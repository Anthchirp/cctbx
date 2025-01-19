from sfapi_client         import Client, SfApiError
from sfapi_client.compute import Machine

from pathlib import Path

import os
import io
import sys
import json

import logging
LOGGER = logging.getLogger(__name__)
# LOGGER.setLevel(logging.DEBUG)
HANDLER = logging.StreamHandler(sys.stdout)
HANDLER.setFormatter(
    logging.Formatter(
        "[%(levelname)8s | %(filename)s:%(lineno)s] %(message)s"
    )
)
LOGGER.addHandler(HANDLER)


class Singleton(type):
    """
    class MySingleton(metaclass=Singleton):
        ...
    Setting `metaclass=Singleton` in the classes meta descriptor marks it as a
    singleton object: if the object has already been constructed elsewhere in
    the code, subsequent calls to the constructor just return this original
    instance.
    """

    # Stores instances in a dictionary:
    # {class: instance}
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        """
        Metclass __call__ operator is called before the class constructor -- so
        this operator will check if an instance already exists in
        Singleton._instances. If it doesn't call the constructor and add the
        instance to Singleton._instances. If it does, then don't call the
        constructor and return the instance instead.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls]


class KeyManager(metaclass=Singleton):

    def __init__(self):
        """
        Store SFAPI key alongsite basic user data (user profile, home location)
        """
        key_store = Path("~/.superfacility/").expanduser().resolve()

        LOGGER.info(f"Checking SFAPI key at: '{key_store}'")

        if not key_store.is_dir():
            raise RuntimeError(f"Secret store {key_store} does not exist") 

        # Load the client_id for the key
        with open(key_store / "clientid.txt", "r") as f:
            client_id = "".join(f.readlines())

        # Get the path for your json file here
        sfapi_key = key_store / "priv_key.pem"
        # ensure the correct file permissions required by the SFAPI
        if sfapi_key.stat() != 0o600:
            sfapi_key.chmod(0o600)

        # ensure that the client key file starts with the client id:
        with sfapi_key.open("r") as f:
            sfapi_key_file = [x.strip() for x in f.readlines()]

        if sfapi_key_file[0] != client_id:
            with open(sfapi_key, "w") as f:
                f.write(f"{client_id}\n" + "\n".join(sfapi_key_file))

        self._client_id = client_id
        self._sfapi_key = sfapi_key

        LOGGER.debug(f"Collecting user data using key at: '{self.key}'")
        with Client(key=self.key) as client:
            self._user = client.user()
            self._home = f"/global/homes/{self.user.name[0]}/{self.user.name}/"

        LOGGER.info(f"Key corresponding to user: '{self.user.name}' is OK")

    @property
    def id(self):
        return self._client_id

    @property
    def key(self):
        return self._sfapi_key

    @property
    def user(self):
        return self._user

    @property
    def home(self):
        return self._home


class SFAPIFile(io.StringIO):
    def __init__(self, path):
        self.path     = path
        self.dirname  = os.path.dirname(path)
        self.filename = os.path.basename(path)

        super().__init__()

    def back_to_start(self):
        self.seek(0)
        bio = io.BytesIO(self.read().encode("utf8"))
        bio.dirname  = self.dirname
        bio.filename = self.filename
        return bio

    def set_data(self, data):
        old_data = self.back_to_start()
        self.write(data)
        # self.seek(0)
        return old_data


class OpenSFAPI:
    def __init__(self, path, mode, mk_target_dir=False):

        valid_mode_chars = set("rwab")
        input_mode_chars = set(mode)

        # If we have duplicate raise exception
        if len(input_mode_chars) != len(mode):
            raise ValueError(f"invalid mode: '{mode}'")

        # check mode chars
        if not input_mode_chars.issubset(valid_mode_chars):
            raise ValueError(f"invalid mode: '{mode}'")

        self._km = KeyManager()

        LOGGER.debug(
            f"Intiating SFAPI connection using key file: {self._km.key}"
        )

        self.client  = Client(key=self._km.key)
        self.compute = self.client.compute(Machine.perlmutter)
        self.buffer  = SFAPIFile(path.replace("~/", self._km.home))

        if "w" in input_mode_chars or "a" in input_mode_chars:
            self.write_mode = True
        else:
            self.write_mode = False

        if "r" in input_mode_chars or "a" in input_mode_chars:
            self.read_mode = True

            LOGGER.debug(f"Getting remote path handle to: {self.buffer.path}")
            try:
                [file] = self.compute.ls(self.buffer.path, directory=False)
            except SfApiError:
                file = None

            if file is not None:
                LOGGER.info(f"Downloading data from: {self.buffer.path}")
                binary = "b" in input_mode_chars
                data   = file.download(binary).read()
                if binary:
                    self.buffer.set_data(data.decode("utf8"))
                else:
                    self.buffer.set_data(data)
                if "a" not in input_mode_chars:
                    self.buffer.seek(0)
        else:
            self.read_mode = False

        LOGGER.info((
            "Initiated SFAPI connection for: "
            f"user '{self._km.user.name}' on machine '{self.compute.name}'"
        ))

        self.mk_target_dir = False if "r" in input_mode_chars else mk_target_dir

    def __enter__(self):
        return self.buffer

    def close(self):
        if not self.write_mode:
            self.client.close()
            LOGGER.debug("File not opened for writing => skipping upload")
            return

        LOGGER.info(f"Writing file to remote location: {self.buffer.path}")
        LOGGER.debug(f"The machine is: {self.compute.status}")

        if self.mk_target_dir:
            LOGGER.debug(f"Ensuring that {self.buffer.dirname} exists")
            self.compute.run(f"mkdir -p {self.buffer.dirname}")

        LOGGER.debug(f"Getting remote path handle to: {self.buffer.dirname}")
        [dir] = self.compute.ls(self.buffer.dirname, directory=True)

        LOGGER.debug(f"Uploading data to: {self.buffer.filename}")
        dir.upload(self.buffer.back_to_start())

        self.client.close()
        LOGGER.debug("DONE")

    def __exit__(self, type, value, traceback):
        self.close()


class PathSFAPI:
    @staticmethod
    def exists(path):
        km = KeyManager()
        target = path.replace("~/", km.home)

        with Client(key=km.key) as client:
            LOGGER.info(f"Attempting to `ls` '{target}'")
            compute = client.compute(Machine.perlmutter)
            LOGGER.debug(f"The machine is: {compute.status}")
            try:
                compute.ls(target)
                LOGGER.debug("Remote `ls` was succesful!")
                target_exists = True
            except SfApiError as e:
                LOGGER.debug(f"Remote `ls` failed with: '{e}'")
                target_exists = False

        return target_exists

    def __getattr__(self, name):
        return getattr(os.path, name)


class OsSFAPI:
    def __init__(self):
        self.path = PathSFAPI()

    @staticmethod
    def open(
        file,
        mode='r', buffering=-1, encoding=None, errors=None, newline=None,
        closefd=True, opener=None, mk_target_dir=False
    ):
        return OpenSFAPI(file, mode, mk_target_dir=mk_target_dir)

    @staticmethod
    def mkdir(path, mode=0o777, *, dir_fd=None): 
        km = KeyManager()
        target = path.replace("~/", km.home)

        with Client(key=km.key) as client:
            compute = client.compute(Machine.perlmutter)
            arg = f"{json.dumps(target)}, mode={mode}, dir_fd={dir_fd}"
            cmd = json.dumps(f"import os; os.mkdir({arg})")

            LOGGER.info(f"Running: {cmd} on '{compute.name}'")
            out = compute.run(f"python3 -c {cmd}")
            LOGGER.debug(f"Result: {out}")

    @staticmethod
    def makedirs(name, mode=0o777, exist_ok=False):
        km = KeyManager()
        target = name.replace("~/", km.home)

        with Client(key=km.key) as client:
            compute = client.compute(Machine.perlmutter)
            arg = f"{json.dumps(target)}, mode={mode}, exist_ok={exist_ok}"
            cmd = json.dumps(f"import os; os.makedirs({arg})")

            LOGGER.info(f"Running: {cmd} on '{compute.name}'")
            out = compute.run(f"python3 -c {cmd}")
            LOGGER.debug(f"Result: {out}")

    @staticmethod
    def stat(fd):
        km = KeyManager()
        target = fd.replace("~/", km.home)

        with Client(key=km.key) as client:
            compute = client.compute(Machine.perlmutter)
            arg = f"{json.dumps(target)}"
            cmd = json.dumps(
                f"import os; import json; print(json.dumps(os.stat({arg})))"
            )

            LOGGER.info(f"Running: {cmd} on '{compute.name}'")
            out = compute.run(f"python3 -c {cmd}")
            remote_stat = os.stat_result(json.loads(out))
            LOGGER.debug(f"Result: {remote_stat}")

            return remote_stat

    @staticmethod
    def chmod(path, mode, *, dir_fd=None, follow_symlinks=True):
        km = KeyManager()
        target = path.replace("~/", km.home)

        with Client(key=km.key) as client:
            compute = client.compute(Machine.perlmutter)
            arg = (
                    f"{json.dumps(target)}, "
                    f"{oct(mode)}, "
                    f"dir_fd={dir_fd}, "
                    f"follow_symlinks={follow_symlinks}"
            )
            cmd = json.dumps(f"import os; os.chmod({arg})")

            LOGGER.info(f"Running: {cmd} on '{compute.name}'")
            out = compute.run(f"python3 -c {cmd}")
            LOGGER.debug(f"Result: {out}")

    def __getattr__(self, name):
        return getattr(os, name)


class OsWrapper(metaclass=Singleton):
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

