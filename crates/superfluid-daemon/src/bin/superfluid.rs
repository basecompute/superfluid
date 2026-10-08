//! `superfluid serve` — the daemon binary.

use std::path::PathBuf;

use superfluid_daemon::api;
use superfluid_daemon::runtime_pick::RuntimeId;
use superfluid_daemon::serve::SUN_PATH_LEN;

fn unknown_command(c: &str) -> ! {
    if c.starts_with('-') {
        eprintln!("superfluid: {c} needs a command first, e.g. superfluid serve {c} ...");
    } else {
        let hint = superfluid_daemon::cli::suggest(c, superfluid_daemon::cli::COMMANDS.iter().copied())
            .map(|s| format!(" (did you mean {s}?)"))
            .unwrap_or_default();
        eprintln!("superfluid: no command {c:?}{hint}");
    }
    eprintln!("Run 'superfluid --help' for the commands.");
    std::process::exit(2);
}

fn connect(socket: &std::path::Path) -> api::NativeClient {
    api::NativeClient::connect(socket).unwrap_or_else(|e| {
        eprintln!(
            "superfluid: cannot reach a server on {}: {e} (is `superfluid serve` running? --socket names another)",
            socket.display()
        );
        std::process::exit(1);
    })
}

fn default_socket() -> PathBuf {
    superfluid_daemon::cli::default_socket(superfluid_daemon::runtimes::Catalog::from_env().home(), SUN_PATH_LEN)
}

fn runtimes_cmd(args: &[String]) {
    let json = args.iter().any(|a| a == "--json");
    let rest: Vec<&String> = args.iter().filter(|a| *a != "--json").collect();
    if rest.len() > 1 || rest.first().is_some_and(|a| a.starts_with('-')) {
        eprint!("{}", superfluid_daemon::cli::RUNTIMES_HELP);
        std::process::exit(2);
    }
    let catalog = superfluid_daemon::runtimes::Catalog::from_env();
    if let Some(model) = rest.first() {
        let (text, value, served) = superfluid_daemon::runtimes_report::model(&catalog, std::path::Path::new(model.as_str()));
        if json {
            println!("{}", serde_json::to_string_pretty(&value).expect("json"));
        } else {
            print!("{text}");
        }
        std::process::exit(if served { 0 } else { 1 });
    }
    let found = superfluid_daemon::runtimes_report::with_installs(&catalog);
    if json {
        let rows = superfluid_daemon::runtimes_report::table_json(&found);
        println!("{}", serde_json::to_string_pretty(&rows).expect("json"));
        return;
    }
    print!("{}", superfluid_daemon::runtimes_report::table_of(&found, &catalog.home().join("runtimes"), Some(&catalog)));
}

fn runtime_cmd(args: &[String]) {
    use superfluid_adapter_kit::install::Request;
    use superfluid_daemon::installs;
    fn usage() -> ! {
        eprint!("{}", superfluid_daemon::cli::RUNTIME_HELP);
        std::process::exit(2);
    }
    fn fail(e: impl std::fmt::Display) -> ! {
        eprintln!("superfluid: {e}");
        std::process::exit(1);
    }
    let (Some(sub), Some(name)) = (args.first(), args.get(1)) else { usage() };
    let Some(id) = RuntimeId::parse(name).filter(|i| i.install().is_none()) else {
        eprintln!("superfluid: '{name}' is not a runtime id (lowercase letters, digits, '.', '-', '_')");
        std::process::exit(2);
    };
    let rest = superfluid_daemon::cli::split_equals(&args[2..], |f| {
        Some(matches!(f, "--version" | "--backend" | "--from" | "--sha256"))
    });
    let rest = rest.as_slice();
    let catalog = superfluid_daemon::runtimes::Catalog::from_env();
    let family = |backend: &str| backend.split(['-', '_']).next().unwrap_or(backend).to_string();
    let carry_out = |installer: &std::path::Path, plans: &installs::Plans| -> installs::Installed {
        let done = installs::install(&catalog, id, installer, &plans.plans).unwrap_or_else(|e| fail(e));
        for (tried, why) in &done.fell_back {
            eprintln!("superfluid: {id} {tried} does not run here ({why}); fell back to the next");
        }
        // The runtime's own version string may carry the library's path; the name says enough.
        let version = done.version.split_whitespace().next().unwrap_or_default();
        println!(
            "superfluid: {id} {} installed ({version}{}){}",
            done.install.name,
            done.device.as_deref().map(|d| format!(", {d}")).unwrap_or_default(),
            if done.replaced { ", replacing the one of that name" } else { "" }
        );
        if let Some(d) = &done.default {
            let pinned = installs::pinned(&catalog, id).is_some();
            if pinned || installs::list(&catalog, id).len() > 1 {
                let how = if pinned { "pinned with `superfluid runtime use`" } else { "the best-ranked" };
                println!("superfluid: {id} serves from {d} by default ({how})");
            }
        }
        done
    };
    match sub.as_str() {
        "install" => {
            let mut req = Request::default();
            let mut dry_run = false;
            let mut it = rest.iter();
            while let Some(flag) = it.next() {
                let mut value = || it.next().cloned().unwrap_or_else(|| usage());
                match flag.as_str() {
                    "--version" => req.version = Some(value()),
                    "--backend" => req.backend = Some(value().to_ascii_lowercase()),
                    "--from" => req.from = Some(value()),
                    "--sha256" => req.sha256 = Some(value().to_ascii_lowercase()),
                    "--untested" => req.untested = true,
                    "--dry-run" => dry_run = true,
                    _ => usage(),
                }
            }
            let installer = installs::installer(&catalog, id).unwrap_or_else(|e| fail(e));
            let plans = installs::plan(&installer, &req).unwrap_or_else(|e| fail(e));
            for (i, p) in plans.plans.iter().enumerate() {
                let size = p.size() + p.extra["wheels_size"].as_u64().unwrap_or(0);
                eprintln!(
                    "superfluid: {} {id} {}: {}{}{}",
                    if i == 0 { "installing" } else { "  then, if it finds no device:" },
                    p.install,
                    p.why,
                    if size > 0 { format!(", {:.1} MB", size as f64 / 1e6) } else { String::new() },
                    if p.tested { String::new() } else { " (UNTESTED with this adapter)".to_string() }
                );
                for a in &p.assets {
                    // The digest is checked either way; it is shown when asked to look before installing.
                    let digest = a.sha256.as_deref().filter(|_| dry_run).map(|s| format!(" (sha256 {s})")).unwrap_or_default();
                    eprintln!("superfluid:   from {}{digest}", a.url);
                }
            }
            if dry_run {
                return;
            }
            carry_out(&installer, &plans);
        }
        "update" => {
            if !rest.is_empty() {
                usage();
            }
            let Some(current) = installs::default(&catalog, id) else {
                fail(format!("{id} is not installed: `superfluid runtime install {id}` installs it"))
            };
            let was_pinned = installs::pinned(&catalog, id).as_deref() == Some(current.name.as_str());
            let installer = installs::installer(&catalog, id).unwrap_or_else(|e| fail(e));
            let req = Request { backend: Some(family(&current.manifest.backend)), ..Request::default() };
            let plans = installs::plan(&installer, &req).unwrap_or_else(|e| fail(e));
            if installs::list(&catalog, id).iter().any(|i| i.name == plans.plans[0].install) {
                println!("{id} {} is installed; nothing newer is tested for {}", plans.plans[0].install, current.manifest.backend);
                return;
            }
            if let Some(why) = installs::not_an_update(&current.manifest.version, &plans.plans[0].version) {
                println!("{id} {}: {why}; nothing to update (`superfluid runtime install {id}` installs {} beside it)", current.name, plans.plans[0].install);
                return;
            }
            let done = carry_out(&installer, &plans);
            if was_pinned {
                installs::pin(&catalog, id, &done.install.name).unwrap_or_else(|e| fail(e));
                println!("{id} serves from {} by default (the pin moved from {})", done.install.name, current.name);
            }
            println!("{id} {} stays installed: `superfluid runtime remove {id} {}` removes it", current.name, current.name);
        }
        "repair" => {
            let target = match rest {
                [] => installs::default(&catalog, id),
                [name] => installs::list(&catalog, id).into_iter().find(|i| i.name == *name),
                _ => usage(),
            }
            .unwrap_or_else(|| fail(format!("{id} has no such install (installed: see `superfluid runtimes`)")));
            let m = &target.manifest;
            let mut req = Request { version: Some(m.version.clone()), backend: Some(m.backend.clone()), untested: !m.tested, ..Request::default() };
            if m.version == "custom" {
                req.from = m.sources.first().map(|(url, _)| url.clone());
                req.sha256 = m.sources.first().map(|(_, sha)| sha.clone()).filter(|s| !s.is_empty());
            }
            let installer = installs::installer(&catalog, id).unwrap_or_else(|e| fail(e));
            let plans = installs::plan(&installer, &req).unwrap_or_else(|e| fail(e));
            carry_out(&installer, &plans);
        }
        "remove" => {
            let name = match rest {
                [] => None,
                [name] => Some(name.as_str()),
                _ => usage(),
            };
            match installs::remove(&catalog, id, name) {
                Ok(gone) => {
                    println!("removed {id} {}", gone.join(", "));
                    if let Some(d) = installs::default(&catalog, id) {
                        println!("{id} serves from {} by default", d.name);
                    }
                }
                Err(e) => fail(e),
            }
        }
        "use" => match rest {
            [flag] if flag == "--auto" => match installs::unpin(&catalog, id) {
                Ok(Some(d)) => println!("{id} serves from {d} by default (the best-ranked)"),
                Ok(None) => fail(format!("{id} is not installed")),
                Err(e) => fail(e),
            },
            [name] => match installs::pin(&catalog, id, name) {
                Ok(()) => println!("{id} serves from {name} by default (pinned; `superfluid runtime use {id} --auto` unpins it)"),
                Err(e) => fail(e),
            },
            _ => usage(),
        },
        _ => usage(),
    }
}

fn session_cmd(args: &[String]) {
    let sub = args.first().map(String::as_str).unwrap_or_else(|| session_usage());
    let id: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| session_usage());
    let mut socket: Option<PathBuf> = None;
    let mut format = String::from("jsonl");
    let mut out_dir: Option<PathBuf> = None;
    let mut include_content = false;
    let mut allow_content_egress = false;
    let mut endpoint: Option<String> = None;
    let mut json = false;
    let mut model = String::new();
    let args = superfluid_daemon::cli::split_equals(args, |f| {
        Some(matches!(f, "--socket" | "--format" | "--out" | "--endpoint" | "--model"))
    });
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| session_usage());
        match a.as_str() {
            "--socket" => socket = Some(PathBuf::from(val())),
            "--format" => format = val(),
            "--out" => out_dir = Some(PathBuf::from(val())),
            "--include-content" => include_content = true,
            "--allow-content-egress" => allow_content_egress = true,
            "--endpoint" => endpoint = Some(val()),
            "--json" => json = true,
            "--model" => model = val(),
            _ => session_usage(),
        }
    }
    let socket = socket.unwrap_or_else(default_socket);
    let mut client = connect(&socket);
    let inspection = match client.request(&api::Request::Inspect { session: id }) {
        Ok(api::Response::Inspection { inspection }) => inspection,
        Ok(api::Response::Err { message }) => {
            eprintln!("superfluid: {message}");
            std::process::exit(1);
        }
        other => {
            eprintln!("superfluid: unexpected reply {other:?}");
            std::process::exit(1);
        }
    };
    match sub {
        "inspect" => {
            if json {
                println!("{}", serde_json::to_string_pretty(&inspection).unwrap());
            } else {
                let s = &inspection.summary;
                println!("session {}  parent {:?}  fork_at {}  title {:?}  archived {}", s.id, s.parent, s.fork_at, s.title, s.archived);
                println!("epoch {}  generation {}  meta_version {}  qos {}  batch_invariant {}", inspection.epoch, s.generation, s.meta_version, inspection.qos_class, inspection.batch_invariant);
                println!("tier {:?}  tokens {}  events {}  last_finish {}  wal_v{}", inspection.tier, s.tokens, s.events, inspection.last_finish, inspection.wal_version);
                if let Some(fp) = inspection.behavior_fingerprint {
                    println!("behavior_fingerprint {fp:016x}");
                }
                if let Some(sid) = inspection.store_id {
                    println!("store_id {}", sid.to_hex());
                }
                for (cid, name, args) in &inspection.open_tool_calls {
                    println!("open tool call {cid}: {name} {args}");
                }
                for e in &inspection.events {
                    let kind = format!("{:?}", e.body);
                    let kind = kind.split([' ', '{']).next().unwrap_or("");
                    println!("  #{:<5} epoch {:<3} ts {:<14} {}", e.event_id, e.epoch, e.ts_unix_ms, kind);
                }
            }
        }
        "export" => {
            let has_trace_ids = endpoint.is_some() || format == "otlp-jsonl";
            let store_ns = match inspection.store_id {
                Some(sid) => superfluid_daemon::export::store_trace_ns(&sid),
                None if !has_trace_ids => 0,
                None => {
                    eprintln!(
                        "superfluid: WARNING the daemon did not report its store id (an older \
                         superfluid): trace ids are un-namespaced — they can collide with other \
                         stores' and will not match live OTLP links"
                    );
                    0
                }
            };
            if let Some(url) = endpoint {
                if include_content && !allow_content_egress {
                    eprintln!(
                        "superfluid: --include-content over --endpoint requires --allow-content-egress; pushing ids-only"
                    );
                }
                let opts = superfluid_daemon::export::ExportOptions {
                    include_content: include_content && allow_content_egress,
                    model,
                    store_ns,
                    ..Default::default()
                };
                eprintln!("superfluid: pushing session {id} traces to {url} (an EGRESS)");
                match superfluid_daemon::export::push_otlp_traces(&url, id, &inspection.events, &opts) {
                    Ok(status) => eprintln!("superfluid: otlp endpoint answered {status}"),
                    Err(e) => {
                        eprintln!("superfluid: {e}");
                        std::process::exit(1);
                    }
                }
                return;
            }
            let opts = superfluid_daemon::export::ExportOptions {
                include_content,
                model,
                store_ns,
                ..Default::default()
            };
            let (name, text) = match format.as_str() {
                "jsonl" => (
                    format!("session-{id}.jsonl"),
                    superfluid_daemon::export::jsonl(id, &inspection.events, &opts),
                ),
                "otlp-jsonl" => (
                    format!("session-{id}.traces.otlp.jsonl"),
                    superfluid_daemon::export::otlp_jsonl(id, &inspection.events, &opts),
                ),
                _ => session_usage(),
            };
            match out_dir {
                Some(dir) => {
                    std::fs::create_dir_all(&dir).expect("create out dir");
                    let path = dir.join(name);
                    let mut o = std::fs::OpenOptions::new();
                    o.create(true).write(true).truncate(true);
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::OpenOptionsExt;
                        o.mode(0o600);
                    }
                    let mut f = o.open(&path).expect("open export file");
                    std::io::Write::write_all(&mut f, text.as_bytes()).expect("write export");
                    eprintln!("superfluid: wrote {}", path.display());
                }
                None => print!("{text}"),
            }
        }
        _ => session_usage(),
    }
}

fn session_usage() -> ! {
    eprint!("{}", superfluid_daemon::cli::SESSION_HELP);
    std::process::exit(2);
}

fn media_usage() -> ! {
    eprint!("{}", superfluid_daemon::cli::MEDIA_HELP);
    std::process::exit(2);
}

fn media_cmd(args: &[String]) {
    let sub = args.first().map(String::as_str).unwrap_or("");
    let mut socket: Option<PathBuf> = None;
    let mut file: Option<PathBuf> = None;
    let args = superfluid_daemon::cli::split_equals(args, |f| Some(f == "--socket"));
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| media_usage());
        match a.as_str() {
            "--socket" => socket = Some(PathBuf::from(val())),
            other if file.is_none() && !other.starts_with("--") => file = Some(PathBuf::from(other)),
            _ => media_usage(),
        }
    }
    let socket = socket.unwrap_or_else(default_socket);
    let mut client = connect(&socket);
    let reply = match sub {
        "put" => {
            let path = file.unwrap_or_else(|| media_usage());
            let bytes = std::fs::read(&path).expect("read media file");
            client.request(&api::Request::PutMedia {
                bytes,
                mime: String::new(),
            })
        }
        "gc" => client.request(&api::Request::GcMedia),
        _ => media_usage(),
    };
    match reply {
        Ok(api::Response::Media { hash }) => println!("{hash}"),
        Ok(api::Response::Gc { kept, removed }) => println!("kept {kept}, removed {removed}"),
        Ok(api::Response::Err { message }) => {
            eprintln!("superfluid: {message}");
            std::process::exit(1);
        }
        other => {
            eprintln!("superfluid: unexpected reply {other:?}");
            std::process::exit(1);
        }
    }
}


fn main() {
    use superfluid_daemon::cli;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str);
    match command {
        None => {
            eprint!("{}", cli::overview());
            std::process::exit(2);
        }
        Some("-h" | "--help") => {
            print!("{}", cli::overview());
            return;
        }
        Some("help") => {
            match args.get(1).map(String::as_str) {
                None => print!("{}", cli::overview()),
                Some(c) => match cli::command_help(c) {
                    Some(h) => print!("{h}"),
                    None => unknown_command(c),
                },
            }
            return;
        }
        Some("-V" | "--version" | "version") => {
            println!("{}", cli::version());
            return;
        }
        Some(c) if c != "serve" && args[1..].iter().take_while(|a| *a != "--").any(|a| a == "--help" || a == "-h") => {
            match cli::command_help(c) {
                Some(h) => print!("{h}"),
                None => unknown_command(c),
            }
            return;
        }
        _ => {}
    }
    if command == Some("launch") {
        superfluid_daemon::launch::run(&args[1..]);
        return;
    }
    if command == Some("stop") {
        superfluid_daemon::launch::stop(&args[1..]);
        return;
    }
    if command == Some("fleet") {
        superfluid_daemon::fleet_token::run(&args[1..]);
        return;
    }
    if command == Some("node") {
        superfluid_daemon::fleet_join::run(&args[1..]);
        return;
    }
    if command == Some("session") {
        session_cmd(&args[1..]);
        return;
    }
    #[cfg(feature = "tui")]
    if command == Some("top") {
        top_cmd(&args[1..]);
        return;
    }
    if command == Some("media") {
        media_cmd(&args[1..]);
        return;
    }
    if command == Some("runtimes") {
        runtimes_cmd(&args[1..]);
        return;
    }
    if command == Some("runtime") {
        runtime_cmd(&args[1..]);
        return;
    }
    if command != Some("serve") {
        unknown_command(command.unwrap_or_default());
    }
    superfluid_daemon::serve::run(&args[1..], &superfluid_daemon::serve::Embedding::default());
}

#[cfg(feature = "tui")]
fn top_cmd(args: &[String]) {
    let usage = || -> ! {
        eprintln!(
            "usage: superfluid top [--api-key <key>] <host:port> [<host:port> ...]   \
             (each a `serve --http` endpoint; the key opens a keyed node's /metrics)"
        );
        std::process::exit(2);
    };
    let mut nodes: Vec<String> = Vec::new();
    let mut api_key: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--api-key" => api_key = Some(it.next().cloned().unwrap_or_else(|| usage())),
            "--help" | "-h" => usage(),
            s if s.starts_with("--") => {
                eprintln!("superfluid top: unknown flag {s}");
                usage();
            }
            s => nodes.push(s.to_string()),
        }
    }
    if nodes.is_empty() {
        usage();
    }
    if let Err(e) = superfluid_daemon::tui_top::run(nodes, api_key) {
        eprintln!("superfluid top: {e}");
        std::process::exit(1);
    }
}
