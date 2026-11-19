//! Ctrl+Break on a Windows console reaches every process on it: the server, which answers it by
//! draining, and the workers the server started. A worker has to live through it, or the
//! generations a drain is there to finish end with the key.

#![cfg(windows)]

use std::io::{BufRead, BufReader, Write};
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use superfluid_daemon::{EngineHost, WorkerSpec};
use windows_sys::Win32::Foundation::BOOL;
use windows_sys::Win32::System::Console::{GenerateConsoleCtrlEvent, SetConsoleCtrlHandler, CTRL_BREAK_EVENT};

/// Set in the test's own program when it runs as the server of
/// `a_worker_lives_through_the_ctrl_break_its_server_drains_on`.
const SERVER_SIDE: &str = "SUPERFLUID_CONSOLE_TEST_SERVER_SIDE";

static KEY_HEARD: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn heard(kind: u32) -> BOOL {
    if kind == CTRL_BREAK_EVENT {
        KEY_HEARD.store(true, Ordering::SeqCst);
    }
    BOOL::from(kind == CTRL_BREAK_EVENT)
}

/// The server: it answers the key itself, starts a worker on the mock engine, says the worker's
/// process id, and once the key has come says whether the worker still runs.
fn server_side() {
    // SAFETY: the handler is a function that lives as long as the process and sets a flag.
    unsafe {
        SetConsoleCtrlHandler(Some(heard), 1);
    }
    let host = EngineHost::spawn_process(WorkerSpec {
        bin: env!("CARGO_BIN_EXE_superfluid-workerd").into(),
        args: vec!["--engine".into(), "mock".into()],
        stderr: None,
    })
    .expect("spawn worker process");
    let pid = host.worker_pid().expect("process backend has a pid");
    // On lines of their own, whatever the test harness has written on the current one.
    println!("\nworker {pid}");
    let by = Instant::now() + Duration::from_secs(20);
    while !KEY_HEARD.load(Ordering::SeqCst) && Instant::now() < by {
        std::thread::sleep(Duration::from_millis(10));
    }
    // The key reached the worker when it reached this process. A worker that takes no notice
    // of it is ended within milliseconds.
    std::thread::sleep(Duration::from_millis(500));
    println!("\nheard {} alive {}", KEY_HEARD.load(Ordering::SeqCst), superfluid_daemon::sys::alive(pid as i32));
    let _ = std::io::stdout().flush();
    drop(host);
}

#[test]
fn a_worker_lives_through_the_ctrl_break_its_server_drains_on() {
    if std::env::var_os(SERVER_SIDE).is_some() {
        return server_side();
    }
    // The key is sent to a console process group, so the server is this test's own program in
    // a group of its own: the test and cargo are on the same console, and are left out.
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    let mut server = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "a_worker_lives_through_the_ctrl_break_its_server_drains_on", "--nocapture"])
        .env(SERVER_SIDE, "1")
        .creation_flags(CREATE_NEW_PROCESS_GROUP)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut said = BufReader::new(server.stdout.take().unwrap()).lines().map_while(Result::ok);
    let worker = said.find(|l| l.starts_with("worker ")).expect("the server started a worker");
    // SAFETY: a console control event for the group the server leads.
    if unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, server.id()) } == 0 {
        let why = std::io::Error::last_os_error();
        let _ = server.kill();
        let _ = server.wait();
        // A process with no console (a service) has no key to send.
        eprintln!("SKIP: no console to send Ctrl+Break on ({why})");
        return;
    }
    let after = said.find(|l| l.starts_with("heard ")).expect("the server said how its worker fared");
    assert!(server.wait().unwrap().success(), "the server's side ended as it should");
    assert_eq!(after, "heard true alive true", "{worker}: Ctrl+Break reached the server, and its worker lived through it");
}
