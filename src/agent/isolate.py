"""Enter a private Linux namespace and exec the sandbox helper.

This process is trusted host code. It must not import the candidate. If setup
fails it exits 96 and the host does not run the candidate anywhere else.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys

# Clone flags from linux/sched.h.
CLONE_NEWNS = 0x00020000
CLONE_NEWUTS = 0x04000000
CLONE_NEWIPC = 0x08000000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000

MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REC = 16384
MS_BIND = 4096
MS_PRIVATE = 1 << 18
MS_REMOUNT = 32
MNT_DETACH = 2

PR_SET_NO_NEW_PRIVS = 38
PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
AUDIT_ARCH_X86_64 = 0xC000003E
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
EPERM = 1

# Namespace-creating clone bits. The bootstrap filter still allows an ordinary
# fork so the trusted supervisor can start one worker, and so pandas can
# initialize. The worker then denies every clone/fork. These byte sizes must
# stay equal to agent.limits; tests compare them.
_NS_MASK = 0x7E020080
_ROOT_BYTES = 1 * 1024 * 1024
_WORK_BYTES = 2 * 1024 * 1024
_TMP_BYTES = 4 * 1024 * 1024
_DEV_BYTES = 1 * 1024 * 1024
_WRITABLE_MOUNTS = {"/work", "/tmp", "/dev"}
_DEVICE_MOUNTS = {"/dev/null", "/dev/zero", "/dev/full", "/dev/urandom"}
# mount, umount2, pivot_root, the new mount API, unshare, setns, ptrace,
# process_vm_readv/writev, socket family, io_uring, module loading, bpf, perf,
# userfaultfd, clone3, open_by_handle_at, pidfd_getfd, keyring, memfd, landlock,
# syslog, reboot, swap, and uid changes. Capabilities are also dropped; this
# list is defense in depth, not a full allowlist.
_BLOCKED = (
    165,
    166,
    155,
    272,
    308,
    101,
    310,
    311,
    41,
    42,
    43,
    49,
    50,
    288,
    44,
    46,
    45,
    47,
    53,
    299,
    307,
    425,
    426,
    427,
    320,
    313,
    175,
    176,
    304,
    321,
    298,
    323,
    435,
    428,
    429,
    430,
    431,
    432,
    442,
    438,
    444,
    445,
    446,
    319,
    103,
    167,
    168,
    169,
    248,
    249,
    250,
    312,
    105,
    106,
    117,
    119,
)

libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_ulong,
    ctypes.c_void_p,
]
libc.mount.restype = ctypes.c_int
libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
libc.umount2.restype = ctypes.c_int
libc.pivot_root.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
libc.pivot_root.restype = ctypes.c_int


def _os_error(prefix: str) -> OSError:
    number = ctypes.get_errno()
    return OSError(number, f"{prefix}: {os.strerror(number)}")


def _mount(source: str | None, target: str, fstype: str | None, flags: int, data: str | None = None) -> None:
    if libc.mount(
        None if source is None else source.encode(),
        target.encode(),
        None if fstype is None else fstype.encode(),
        flags,
        None if data is None else data.encode(),
    ) != 0:
        raise _os_error(f"mount {source} -> {target}")


def _write_id_maps(uid: int, gid: int) -> None:
    with open("/proc/self/setgroups", "w", encoding="utf-8") as handle:
        handle.write("deny")
    with open("/proc/self/uid_map", "w", encoding="utf-8") as handle:
        handle.write(f"0 {uid} 1\n")
    with open("/proc/self/gid_map", "w", encoding="utf-8") as handle:
        handle.write(f"0 {gid} 1\n")


def _remount_readonly(target: str) -> None:
    if libc.mount(None, target.encode(), None, MS_REMOUNT | MS_RDONLY, None) != 0:
        raise _os_error(f"remount read-only {target}")


def _mount_table() -> list[list[str]]:
    rows = []
    with open("/proc/mounts", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 4:
                rows.append(parts)
    return rows


def _seal_and_check_mounts() -> None:
    """Remount every non-allowlisted writable mount read-only, then refuse a bad table.

    `/` must be the tmpfs created above, not the host ext4 directory that
    pivot_root used to keep. A failed seal does not continue into the helper.
    """
    for parts in _mount_table():
        target, options = parts[1], set(parts[3].split(","))
        if "rw" in options and target not in _WRITABLE_MOUNTS and target not in _DEVICE_MOUNTS:
            _remount_readonly(target)
    problems: list[str] = []
    seen: set[str] = set()
    for parts in _mount_table():
        source, target, fstype, option_text = parts[0], parts[1], parts[2], parts[3]
        options = set(option_text.split(","))
        seen.add(target)
        writable = "rw" in options
        if target == "/":
            # The root must be the private tmpfs. A read-only ext4 bind of the
            # host temp directory is not accepted.
            if fstype != "tmpfs" or writable or "ro" not in options:
                problems.append(f"root is {source} {fstype} {option_text}")
        elif target == "/usr" or target.startswith("/usr/"):
            # /usr is the read-only interpreter tree. Its backing device may be
            # the host filesystem; it is still rejected unless the mount is ro.
            if writable or "ro" not in options:
                problems.append(f"usr mount {target} is writable")
        elif target == "/proc" or target.startswith("/proc/"):
            if fstype != "proc" or writable or "ro" not in options:
                problems.append(f"proc mount {target} is {fstype} {option_text}")
        elif target in _WRITABLE_MOUNTS:
            if fstype != "tmpfs" or not writable:
                problems.append(f"{target} is {fstype} {option_text}")
        elif target in _DEVICE_MOUNTS:
            continue
        else:
            problems.append(f"unexpected mount {target} {source} {fstype} {option_text}")
    for required in ("/", "/usr", "/proc", "/work", "/tmp", "/dev"):
        if required not in seen:
            problems.append(f"missing mount {required}")
    if problems:
        raise OSError(1, "mount seal failed: " + "; ".join(problems[:8]))


def _setup_root(work: str, newroot: str) -> None:
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    os.makedirs(newroot, exist_ok=True)
    # Cover the host directory before creating anything inside it. The tmpfs,
    # not the host ext4 inode, becomes the sandbox root.
    _mount("tmpfs", newroot, "tmpfs", MS_NOSUID | MS_NODEV, f"size={_ROOT_BYTES},mode=755")
    for name in ("usr", "proc", "tmp", "dev", "work", "payload", "oldroot"):
        os.makedirs(f"{newroot}/{name}", exist_ok=True)
    _mount("/usr", f"{newroot}/usr", None, MS_BIND | MS_REC)
    _mount(None, f"{newroot}/usr", None, MS_BIND | MS_REMOUNT | MS_RDONLY)
    for link, target in (("lib", "usr/lib"), ("lib64", "usr/lib64"), ("bin", "usr/bin")):
        path = f"{newroot}/{link}"
        if not os.path.lexists(path):
            os.symlink(target, path)
    _mount("proc", f"{newroot}/proc", "proc", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC)
    _mount("tmpfs", f"{newroot}/tmp", "tmpfs", MS_NOSUID | MS_NODEV, f"size={_TMP_BYTES},mode=700")
    _mount("tmpfs", f"{newroot}/dev", "tmpfs", MS_NOSUID | MS_NOEXEC, f"size={_DEV_BYTES},mode=755")
    for node in ("null", "zero", "full", "urandom"):
        destination = f"{newroot}/dev/{node}"
        open(destination, "a", encoding="utf-8").close()
        _mount(f"/dev/{node}", destination, None, MS_BIND)
    # /work is a size-capped tmpfs, not a bind of the host directory.
    _mount("tmpfs", f"{newroot}/work", "tmpfs", MS_NOSUID | MS_NODEV, f"size={_WORK_BYTES},mode=700")
    for name in ("inner.py", "candidate.py", "runtime_failure.txt"):
        source = os.path.join(work, name)
        if name == "runtime_failure.txt" and not os.path.exists(source):
            continue
        target = f"{newroot}/payload/{name}"
        with open(source, "rb") as incoming, open(target, "wb") as outgoing:
            outgoing.write(incoming.read())
        os.chmod(target, 0o644)
    os.chdir(newroot)
    if libc.pivot_root(b".", b"oldroot") != 0:
        raise _os_error("pivot_root")
    os.chdir("/")
    if libc.umount2(b"/oldroot", MNT_DETACH) != 0:
        raise _os_error("umount old root")
    try:
        os.rmdir("/oldroot")
    except OSError as exc:
        raise OSError(exc.errno, "old root is still present after umount") from exc
    _seal_and_check_mounts()


def _set_limits(cpu_seconds: int) -> None:
    import resource

    def limit(name: int, soft: int, hard: int | None = None) -> None:
        ceiling = soft if hard is None else hard
        resource.setrlimit(name, (soft, ceiling))

    limit(resource.RLIMIT_CPU, cpu_seconds)
    limit(resource.RLIMIT_AS, 1024 * 1024 * 1024)
    # Per-file backstop. The aggregate cap is the tmpfs sizes, which a second
    # file under this rlimit can still exhaust.
    limit(resource.RLIMIT_FSIZE, _WORK_BYTES)
    limit(resource.RLIMIT_CORE, 0)
    limit(resource.RLIMIT_NOFILE, 256)
    # RLIMIT_NPROC is intentionally unset. It is accounted for the whole host
    # uid (this environment's ceiling is in the tens of thousands) and would
    # not stop a fork bomb inside the sandbox. The worker denies clone instead.
    # Creating a child cgroup for pids.max is not available: /sys/fs/cgroup is
    # not writable from this process.


def _seccomp_program() -> ctypes.Structure:
    bpf_ld = 0x00
    bpf_w = 0x00
    bpf_abs = 0x20
    bpf_jmp = 0x05
    bpf_jeq = 0x10
    bpf_k = 0x00
    bpf_ret = 0x06
    bpf_alu = 0x04
    bpf_and = 0x50

    def stmt(code: int, k: int) -> bytes:
        return struct.pack("<HBBI", code, 0, 0, k & 0xFFFFFFFF)

    def jump(k: int, jt: int, jf: int) -> bytes:
        return struct.pack("<HBBI", bpf_jmp | bpf_jeq | bpf_k, jt, jf, k & 0xFFFFFFFF)

    blob = b""
    blob += stmt(bpf_ld | bpf_w | bpf_abs, 4)
    blob += jump(AUDIT_ARCH_X86_64, 1, 0)
    blob += stmt(bpf_ret | bpf_k, SECCOMP_RET_KILL_PROCESS)
    blob += stmt(bpf_ld | bpf_w | bpf_abs, 0)
    # clone (56): allow forks/threads that do not create a new namespace.
    blob += jump(56, 0, 5)
    blob += stmt(bpf_ld | bpf_w | bpf_abs, 16)
    blob += stmt(bpf_alu | bpf_and | bpf_k, _NS_MASK)
    blob += jump(0, 1, 0)
    blob += stmt(bpf_ret | bpf_k, SECCOMP_RET_ERRNO | EPERM)
    blob += stmt(bpf_ret | bpf_k, SECCOMP_RET_ALLOW)
    for number in _BLOCKED:
        blob += jump(number, 0, 1)
        blob += stmt(bpf_ret | bpf_k, SECCOMP_RET_ERRNO | EPERM)
    blob += stmt(bpf_ret | bpf_k, SECCOMP_RET_ALLOW)

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint16),
            ("jt", ctypes.c_uint8),
            ("jf", ctypes.c_uint8),
            ("k", ctypes.c_uint32),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    count = len(blob) // 8
    filters = (SockFilter * count).from_buffer_copy(blob)
    program = SockFprog(count, ctypes.cast(filters, ctypes.POINTER(SockFilter)))
    # Keep the buffer alive for the caller.
    program._filters = filters  # type: ignore[attr-defined]
    return program


def _install_seccomp() -> None:
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise _os_error("PR_SET_NO_NEW_PRIVS")
    program = _seccomp_program()
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program)) != 0:
        raise _os_error("seccomp")


def _drop_capabilities() -> None:
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    header = Header(0x20080522, 0)
    data = (Data * 2)()
    if libc.capset(ctypes.byref(header), data) != 0:
        raise _os_error("capset")
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise _os_error("PR_SET_DUMPABLE")


def _close_extra_fds() -> None:
    for name in os.listdir("/proc/self/fd"):
        number = int(name)
        if number > 2:
            try:
                os.close(number)
            except OSError:
                pass


def _exit_code(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return 1


def _become(work: str, newroot: str, cpu_seconds: int, uid: int, gid: int) -> None:
    os.unshare(CLONE_NEWUSER)
    _write_id_maps(uid, gid)
    os.unshare(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWIPC | CLONE_NEWUTS)
    pid = os.fork()
    if pid != 0:
        _pid, status = os.waitpid(pid, 0)
        os._exit(_exit_code(status))
    _setup_root(work, newroot)
    _set_limits(cpu_seconds)
    _install_seccomp()
    _drop_capabilities()
    _close_extra_fds()
    os.chdir("/work")
    env = {
        "PATH": "/usr/bin",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    os.execve("/usr/bin/python3", ["/usr/bin/python3", "-I", "-B", "/payload/inner.py"], env)
    os._exit(127)


def main() -> None:
    if len(sys.argv) != 4:
        sys.stderr.write("isolate usage: isolate.py <work> <newroot> <cpu-seconds>\n")
        os._exit(96)
    work, newroot, cpu_text = sys.argv[1:]
    try:
        cpu_seconds = int(cpu_text)
    except ValueError:
        sys.stderr.write("cpu limit must be an integer\n")
        os._exit(96)
    if cpu_seconds < 1 or cpu_seconds > 180:
        sys.stderr.write("cpu limit is outside the allowed range\n")
        os._exit(96)
    uid, gid = os.getuid(), os.getgid()
    try:
        _become(work, newroot, cpu_seconds, uid, gid)
    except Exception as exc:
        sys.stderr.write(f"namespace setup failed: {exc}\n")
        os._exit(96)


if __name__ == "__main__":
    main()
