//! `superfluid-workerd`, the worker process of a runtime linked into this
//! build, as a library function so that another program can be the worker.

use superfluid_adapter_kit::{Runtime, WorkerArgs};

fn usage(name: &str, why: &str) -> ! {
    eprintln!("{name}: {why}");
    eprintln!(
        "usage: {name} --frames-fd N --fd-channel-fd M --engine mock|native|llamacpp|mlx \
         [--model <path>] [--max-context N] [--max-batch N] [--kv-bits N] [--venv <env>] [--basert-lib <path>]\n       \
         {name} check --engine native|llamacpp|mlx [--basert-lib <path>] [--model <path>]"
    );
    std::process::exit(2);
}

fn runtime(name: &str, args: &WorkerArgs, default: Option<&str>) -> &'static dyn Runtime {
    let Some(engine) = args.extra("--engine").or(default) else { usage(name, "check needs --engine") };
    if let Some(runtime) = crate::linked::engine(engine) {
        return runtime;
    }
    match crate::linked::feature_for(engine) {
        Some(feature) => {
            eprintln!("{name}: built without the {feature} feature");
            std::process::exit(2);
        }
        None => usage(name, &format!("unknown engine {engine:?}")),
    }
}

/// Whether `argv` starts a linked runtime's worker: `--engine <id> ...` or
/// `check --engine <id> ...`.
pub fn is_invocation(argv: &[String]) -> bool {
    match argv.first().map(String::as_str) {
        Some("--engine") => true,
        Some("check") => argv.get(1).map(String::as_str) == Some("--engine"),
        _ => false,
    }
}

/// Runs the worker. `argv` is its command line after the program name;
/// `name` is the program name its messages carry.
pub fn main(argv: Vec<String>, name: &str) -> ! {
    line_buffer_stdout();
    let checking = argv.first().map(String::as_str) == Some("check");
    let args = WorkerArgs::parse(if checking { &argv[1..] } else { &argv[..] }).unwrap_or_else(|e| usage(name, &e));
    let runtime = runtime(name, &args, if checking { None } else { Some("mock") });
    let reads: Vec<&str> = ["--engine"].into_iter().chain(runtime.flags().iter().copied()).collect();
    args.refuse_unread(&reads).unwrap_or_else(|e| usage(name, &e));
    runtime.prepare(&args);
    if checking {
        superfluid_adapter_kit::check(runtime, &args).finish();
    }
    superfluid_adapter_kit::serve_process(runtime, args, name)
}

fn line_buffer_stdout() {
    extern "C" {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        #[link_name = "__stdoutp"]
        static mut C_STDOUT: *mut libc::FILE;
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        #[link_name = "stdout"]
        static mut C_STDOUT: *mut libc::FILE;
    }
    // SAFETY: setvbuf on the process's own stdout before anything has
    // written to it; a null buffer lets libc allocate.
    unsafe {
        libc::setvbuf(C_STDOUT, std::ptr::null_mut(), libc::_IOLBF, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::is_invocation;

    fn line(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn a_workers_command_line_is_told_apart_from_its_hosts() {
        assert!(is_invocation(&line(&["--engine", "basert", "--model", "m.base", "--frames-fd", "3", "--fd-channel-fd", "4"])));
        assert!(is_invocation(&line(&["check", "--engine", "basert", "--model", "m.base"])));
        for host in [&["m.base", "--port", "8453"][..], &["--model", "m.base"], &["check"], &["check", "--model", "m.base"], &[]] {
            assert!(!is_invocation(&line(host)), "{host:?}");
        }
    }
}
