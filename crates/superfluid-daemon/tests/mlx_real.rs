#![cfg(feature = "mlx")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use superfluid_daemon::{BundleCodec, Daemon, EngineHost, SessionStore, WorkerSpec};
use superfluid_engine::Tokenizer;
use superfluid_tokenizer_hf::HfTokenizer;

fn mlx_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SUPERFLUID_TEST_MLX") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let p = root.join("models/Qwen3-0.6B-mlx-4bit");
    p.join("config.json").is_file().then_some(p)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-mlx-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn spawn_mlx_process(model: &Path) -> EngineHost {
    EngineHost::spawn_process(WorkerSpec {
        bin: env!("CARGO_BIN_EXE_superfluid-workerd").into(),
        args: vec![
            "--engine".into(),
            "mlx".into(),
            "--model".into(),
            model.to_string_lossy().into_owned(),
            "--max-context".into(),
            "2048".into(),
            "--max-batch".into(),
            "4".into(),
        ],
        stderr: None,
    })
    .expect("spawn mlx worker process")
}

#[test]
fn worker_process_serves_an_mlx_model_and_resumes_warm_after_a_crash() {
    let Some(model) = mlx_path() else {
        eprintln!("SKIP: no test MLX model");
        return;
    };
    let dir = scratch("proc");
    let park = dir.join("park");
    let host = spawn_mlx_process(&model);
    let pid = host.worker_pid().expect("process backend");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let tok = Arc::new(HfTokenizer::load(&model).expect("tokenizer load"));
    let codec = BundleCodec::from_tokenizer(tok.clone());
    let daemon = Daemon::with_park(store, host, Box::new(codec), 4, Some(park.clone()));

    let session = daemon.create(None, Default::default()).unwrap();
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(6) + "The capital of France is";
    let prompt = tok.encode(&text);
    assert!(prompt.len() >= 32, "prompt is {} tokens", prompt.len());
    daemon.append(session, None, prompt).unwrap();
    let first = daemon.generate(session, 8).unwrap();
    assert!(first.tokens_generated >= 1);
    assert_eq!(first.warm_prefix, 0, "first generation is cold");
    {
        let artifact = park.join(format!("{session}.park"));
        let mut ok = artifact.exists();
        for _ in 0..500 {
            if ok {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            ok = artifact.exists();
        }
        assert!(ok, "retire parked a sealed artifact (background writer)");
    }

    let _ = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .expect("kill");
    let failed = daemon.generate(session, 4);
    assert!(failed.is_err(), "a murdered engine fails typed");

    let out = daemon.generate(session, 4).unwrap();
    assert!(out.tokens_generated >= 1);
    assert!(
        out.warm_prefix >= 16,
        "resume after worker murder is warm from the park artifact (got {})",
        out.warm_prefix
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_preempted_call_resumes_by_decode_replay_and_matches_the_uncontended_run() {
    use superfluid_daemon::qos;
    use std::sync::atomic::Ordering::Relaxed;
    let Some(model) = mlx_path() else {
        eprintln!("SKIP: no test MLX model");
        return;
    };
    let tok = Arc::new(HfTokenizer::load(&model).expect("tokenizer load"));
    let text = "Count upward in words, one number on each line.\none\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n\
                eleven\ntwelve\nthirteen\nfourteen\nfifteen\nsixteen\nseventeen\neighteen\nnineteen\ntwenty\ntwenty-one\n\
                twenty-two\ntwenty-three\ntwenty-four\ntwenty-five\ntwenty-six\ntwenty-seven\ntwenty-eight\ntwenty-nine\n";
    let mut prompt = tok.encode(text);
    assert!(prompt.len() >= 64, "prompt is {} tokens", prompt.len());
    prompt.truncate(64);
    const TOKENS: u32 = 160;
    let serve = |tag: &str| -> (Arc<Daemon>, PathBuf) {
        let dir = scratch(tag);
        let store = SessionStore::open(&dir.join("wal.log")).unwrap();
        let codec = BundleCodec::from_tokenizer(tok.clone());
        let opts = superfluid_daemon::DaemonOptions { max_lanes: 1, tick_target_ms: 40, ..Default::default() };
        (Arc::new(Daemon::with_options(store, spawn_mlx_process(&model), Box::new(codec), opts).unwrap()), dir)
    };
    let tokens_of = |d: &Daemon, s: u64| d.store().lock().unwrap().session(s).unwrap().tokens.clone();

    let (alone, dir_alone) = serve("replay-alone");
    let s = alone.create(None, Default::default()).unwrap();
    alone.append(s, None, prompt.clone()).unwrap();
    let uncontended = alone.generate(s, TOKENS).unwrap();
    let want = tokens_of(&alone, s);
    assert!(uncontended.tokens_generated >= 48, "a call long enough to preempt ({} tokens)", uncontended.tokens_generated);
    alone.shutdown();
    drop(alone);

    let (d, dir) = serve("replay-preempted");
    let stats = d.sched_stats();
    let agent = d.create(None, Default::default()).unwrap();
    d.append(agent, None, prompt.clone()).unwrap();
    d.set_qos(agent, qos::BACKGROUND_AGENT, false).unwrap();
    let call = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || d.generate(agent, TOKENS))
    };
    let mut produced = 0;
    for _ in 0..30_000 {
        produced = stats.decode_tokens.load(Relaxed);
        if produced >= 16 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(produced >= 16, "the agent is decoding");
    let chat = d.create(None, Default::default()).unwrap();
    d.append(chat, None, tok.encode("The capital of France is")).unwrap();
    d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
    let answered = d.generate(chat, 8).expect("the chat is served");
    assert!(answered.tokens_generated >= 1);
    let out = call.join().unwrap().expect("the preempted call completes");
    assert!(stats.preemptions.load(Relaxed) >= 1, "the chat preempted the agent");
    assert_eq!(out.tokens_generated, uncontended.tokens_generated);
    assert_eq!(tokens_of(&d, agent), want, "preempt and resume is transparent");
    d.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir_alone);
}

fn http(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(120))).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("split");
    let status: u16 = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        dechunk(payload)
    } else {
        payload.to_string()
    };
    (status, body)
}

fn dechunk(payload: &str) -> String {
    let mut out = String::new();
    let mut rest = payload;
    while let Some((size_line, tail)) = rest.split_once("\r\n") {
        let Ok(size) = usize::from_str_radix(size_line.trim(), 16) else {
            break;
        };
        if size == 0 || tail.len() < size {
            out.push_str(&tail[..tail.len().min(size)]);
            break;
        }
        out.push_str(&tail[..size]);
        rest = tail[size..].strip_prefix("\r\n").unwrap_or(&tail[size..]);
    }
    out
}

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn serve_binary_answers_the_openai_api_with_runtime_mlx() {
    let Some(model) = mlx_path() else {
        eprintln!("SKIP: no test MLX model");
        return;
    };
    let dir = scratch("serve");
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let child = Child(
        std::process::Command::new(env!("CARGO_BIN_EXE_superfluid"))
            .args([
                "serve",
                "--runtime",
                "mlx",
                "--model",
                &model.to_string_lossy(),
                "--sessions",
                &dir.join("sessions").to_string_lossy(),
                "--socket",
                &dir.join("superfluid.sock").to_string_lossy(),
                "--http",
                &addr.to_string(),
                "--max-context",
                "2048",
                "--max-batch",
                "4",
                "--no-tui",
            ])
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("spawn superfluid serve"),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    let models = loop {
        if let Ok(mut s) = TcpStream::connect(addr) {
            drop(s.write_all(b""));
            let (status, body) = http(addr, "GET", "/v1/models", "");
            if status == 200 {
                break body;
            }
        }
        assert!(std::time::Instant::now() < deadline, "superfluid serve did not come up");
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    let stem = model.file_name().unwrap().to_string_lossy().into_owned();
    assert!(models.contains(&stem), "/v1/models lists the MLX model by its directory name: {models}");

    let (status, body) = http(
        addr,
        "POST",
        "/v1/completions",
        &format!(
            r#"{{"model":"{stem}","prompt":"The capital of France is","max_tokens":8,"temperature":0}}"#
        ),
    );
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("Paris"), "greedy completion names Paris: {body}");

    let (status, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        &format!(
            r#"{{"model":"{stem}","messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":16,"temperature":0}}"#
        ),
    );
    assert_eq!(status, 200, "{body}");
    assert!(body.contains(r#""finish_reason""#), "{body}");

    let (status, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        &format!(
            r#"{{"model":"{stem}","messages":[{{"role":"user","content":"Is the sky blue? Answer as JSON."}}],"max_tokens":48,"temperature":0,"response_format":{{"type":"json_schema","json_schema":{{"name":"x","schema":{{"type":"object","properties":{{"ok":{{"type":"boolean"}}}},"required":["ok"],"additionalProperties":false}}}}}}}}"#
        ),
    );
    assert_eq!(status, 200, "{body}");
    let reply: serde_json::Value = serde_json::from_str(&body).expect("a JSON body");
    let content = reply["choices"][0]["message"]["content"].as_str().expect("content");
    let answer: serde_json::Value = serde_json::from_str(content.trim())
        .unwrap_or_else(|e| panic!("the reply is the schema's value ({e}): {content:?}"));
    assert!(answer["ok"].is_boolean(), "{answer}");
    let (status, _) = http(addr, "GET", "/v1/models", "");
    assert_eq!(status, 200);
    drop(child);
    let _ = std::fs::remove_dir_all(&dir);
}

fn serve_on(model: &Path, dir: &Path, extra: &[&str]) -> (Child, std::net::SocketAddr) {
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut args: Vec<String> = [
        "serve",
        "--runtime",
        "mlx",
        "--model",
        &model.to_string_lossy(),
        "--sessions",
        &dir.join("sessions").to_string_lossy(),
        "--socket",
        &dir.join("superfluid.sock").to_string_lossy(),
        "--http",
        &addr.to_string(),
        "--max-context",
        "2048",
        "--max-batch",
        "4",
        "--no-tui",
    ]
    .iter()
    .map(|a| a.to_string())
    .collect();
    args.extend(extra.iter().map(|a| a.to_string()));
    let child = Child(
        std::process::Command::new(env!("CARGO_BIN_EXE_superfluid"))
            .args(&args)
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("spawn superfluid serve"),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    loop {
        if TcpStream::connect(addr).is_ok() && http(addr, "GET", "/v1/models", "").0 == 200 {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "superfluid serve did not come up");
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    (child, addr)
}

fn metric(addr: std::net::SocketAddr, name: &str) -> f64 {
    let (_, body) = http(addr, "GET", "/metrics", "");
    body.lines()
        .filter(|l| l.starts_with(name) && !l.starts_with('#'))
        .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
        .sum()
}

#[test]
fn serve_binary_speculates_by_prompt_lookup_and_answers_as_plain_decoding_does() {
    let Some(model) = mlx_path() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let stem = model.file_name().unwrap().to_string_lossy().into_owned();
    let body = format!(
        r#"{{"model":"{stem}","prompt":"Copy the list below three times.\n1. alpha\n2. beta\n3. gamma\n4. delta\n5. epsilon\n\n1. alpha\n2. beta\n3. gamma\n4. delta\n5. epsilon\n\n1. alpha\n","max_tokens":48,"temperature":0}}"#
    );
    let text = |reply: &str| -> String {
        let v: serde_json::Value = serde_json::from_str(reply).expect("a JSON body");
        v["choices"][0]["text"].as_str().expect("text").to_string()
    };
    let dir = scratch("spec-plain");
    let (plain, addr) = serve_on(&model, &dir, &[]);
    let (status, reply) = http(addr, "POST", "/v1/completions", &body);
    assert_eq!(status, 200, "{reply}");
    let want = text(&reply);
    drop(plain);
    let _ = std::fs::remove_dir_all(&dir);

    let dir = scratch("spec");
    let (spec, addr) = serve_on(&model, &dir, &["--speculate", "prompt-lookup"]);
    let (status, reply) = http(addr, "POST", "/v1/completions", &body);
    assert_eq!(status, 200, "{reply}");
    assert_eq!(text(&reply), want, "drafts verified against the model are what it would have said");
    let proposed = metric(addr, "superfluid_spec_proposed_total");
    let accepted = metric(addr, "superfluid_spec_accepted_total");
    assert!(proposed > 0.0 && accepted > 0.0, "the list repeats: drafts proposed {proposed}, accepted {accepted}");
    let (_, runtimes) = http(addr, "GET", "/v1/models", "");
    assert!(runtimes.contains(&stem), "{runtimes}");
    drop(spec);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn serve_binary_refuses_a_wrong_pick_and_step1_gaps_typed() {
    let dir = scratch("refuse");
    let fake = dir.join("Fake-mlx");
    std::fs::create_dir_all(&fake).unwrap();
    std::fs::write(fake.join("config.json"), b"{}").unwrap();
    std::fs::write(fake.join("model.safetensors"), b"").unwrap();
    let fake = fake.to_string_lossy().into_owned();
    let run = |extra: &[&str]| -> (i32, String) {
        let mut args = vec![
            "serve",
            "--model",
            fake.as_str(),
            "--sessions",
            "/nonexistent/sessions",
            "--socket",
            "/nonexistent/superfluid.sock",
            "--no-http",
        ];
        args.extend_from_slice(extra);
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_superfluid"))
            .args(&args)
            .current_dir(&dir)
            .output()
            .expect("run superfluid");
        (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    let (code, err) = run(&["--runtime", "basert"]);
    assert_eq!(code, 2, "{err}");
    if Path::new(env!("CARGO_BIN_EXE_superfluid")).with_file_name("superfluid-worker-basert").is_file() {
        assert!(err.contains("runtime basert cannot serve") && err.contains("it reads a .base bundle"), "{err}");
        assert!(err.contains("pass --runtime mlx"), "{err}");
    } else {
        assert!(err.contains("no runtime 'basert' here to serve"), "{err}");
        assert!(err.contains("it is an MLX model directory, which mlx here reads: pass --runtime mlx"), "{err}");
    }
    let (code, err) = run(&["--runtime", "mlx", "--park-lossy"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("--park-lossy: the mlx runtime exports no lossy encoding"), "{err}");
    let (code, err) = run(&["--runtime", "mlx", "--kv-bits", "8"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("--kv-bits 8: the mlx runtime keeps its KV type per context"), "{err}");
    let (code, err) = run(&["--runtime", "vllm"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("no runtime 'vllm' here to serve"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
