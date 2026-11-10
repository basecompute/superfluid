//! `superfluid-workerd`, the worker process of a runtime linked into this
//! build, as a library function so that another program can be the worker.

use superfluid_adapter_kit::{Runtime, WorkerArgs};

fn usage(name: &str, why: &str) -> ! {
    eprintln!("{name}: {why}");
    eprintln!(
        "usage: {name} --frames-fd N --fd-channel-fd M --engine mock|native|llamacpp|mlx \
         [--model <path>] [--max-context N] [--max-batch N] [--kv-bits N] [--venv <env>] [--basert-lib <path>] [--<option> <value>]...\n       \
         {name} check --engine native|llamacpp|mlx [--basert-lib <path>] [--model <path>]\n       \
         {name} ahead --engine native|llamacpp|mlx [--basert-lib <path>] --model <path> [--max-context N]"
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

/// Whether `argv` starts a linked runtime's worker: `--engine <id> ...`,
/// `check --engine <id> ...` or `ahead --engine <id> ...`.
pub fn is_invocation(argv: &[String]) -> bool {
    match argv.first().map(String::as_str) {
        Some("--engine") => true,
        Some("check" | "ahead") => argv.get(1).map(String::as_str) == Some("--engine"),
        _ => false,
    }
}

/// Runs the worker. `argv` is its command line after the program name;
/// `name` is the program name its messages carry.
pub fn main(argv: Vec<String>, name: &str) -> ! {
    line_buffer_stdout();
    let checking = argv.first().map(String::as_str) == Some("check");
    // `ahead`: the work a runtime does for a model ahead of its first load, in this executable, the one that loads it.
    let ahead = argv.first().map(String::as_str) == Some("ahead");
    let args = if ahead { ahead_args(&argv[1..]) } else { WorkerArgs::parse(if checking { &argv[1..] } else { &argv[..] }) }
        .unwrap_or_else(|e| usage(name, &e));
    let runtime = runtime(name, &args, if checking || ahead { None } else { Some("mock") });
    let reads: Vec<String> = std::iter::once("--engine".to_string()).chain(superfluid_adapter_kit::worker_flags(runtime)).collect();
    args.refuse_unread(&reads.iter().map(String::as_str).collect::<Vec<_>>()).unwrap_or_else(|e| usage(name, &e));
    superfluid_adapter_kit::refuse_bad_options(runtime, &args).unwrap_or_else(|e| usage(name, &e));
    runtime.prepare(&args);
    if ahead {
        let model = args.model.as_deref().map(|m| m.display().to_string()).unwrap_or_default();
        match runtime.ahead(&args) {
            Ok(Some(done)) => eprintln!("superfluid: {model}: {done}"),
            Ok(None) => {}
            Err(why) => {
                eprintln!("{name}: {model}: {why}");
                std::process::exit(1);
            }
        }
        std::process::exit(0);
    }
    if checking {
        superfluid_adapter_kit::check(runtime, &args).finish();
    }
    superfluid_adapter_kit::serve_process(runtime, args, name)
}

/// `ahead`'s arguments. `--max-context` is the window the model will be served at, as a load takes it; without one
/// (a pull's `ahead`, before any serve picks one) it is 0, which the runtime takes as each window a serve can pick,
/// where a worker that serves takes 4096.
fn ahead_args(rest: &[String]) -> Result<WorkerArgs, String> {
    let mut args = WorkerArgs::parse(rest)?;
    if !rest.chunks(2).any(|flag| flag[0] == "--max-context") {
        args.max_context = 0;
    }
    Ok(args)
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
    use super::{ahead_args, is_invocation};

    fn line(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn a_workers_command_line_is_told_apart_from_its_hosts() {
        assert!(is_invocation(&line(&["--engine", "basert", "--model", "m.base", "--frames-fd", "3", "--fd-channel-fd", "4"])));
        assert!(is_invocation(&line(&["check", "--engine", "basert", "--model", "m.base"])));
        assert!(is_invocation(&line(&["ahead", "--engine", "basert", "--model", "m.base"])));
        for host in [&["m.base", "--port", "8453"][..], &["--model", "m.base"], &["check"], &["check", "--model", "m.base"], &["ahead", "--model", "m.base"], &[]] {
            assert!(!is_invocation(&line(host)), "{host:?}");
        }
    }

    #[test]
    fn ahead_prepares_for_the_window_given_or_for_any_without_one() {
        let at = |args: &[&str]| ahead_args(&line(args)).unwrap().max_context;
        assert_eq!(at(&["--engine", "basert", "--model", "m.base", "--max-context", "32768"]), 32768);
        assert_eq!(at(&["--engine", "basert", "--model", "m.base", "--max-context", "4096"]), 4096);
        assert_eq!(at(&["--engine", "basert", "--model", "m.base"]), 0);
    }
}
