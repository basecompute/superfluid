//! `superfluid top` — a multi-node fleet monitor. A fleet head's address stands for every node
//! joined to it: the head's `/metrics` lists them, and the rows follow as nodes come and go.

#![cfg(feature = "tui")]

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::crossterm::{execute, ExecutableCommand};
use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Sparkline, Table};
use ratatui::{Frame, Terminal};

const ACCENT: Color = Color::Cyan;
const HIST: usize = 160;

#[derive(Clone, Default)]
struct NodeStat {
    addr: String,
    host: String,
    reachable: bool,
    unauthorized: bool,
    lanes: u64,
    pool_used: u64,
    pool_total: u64,
    queue: u64,
    tps: u64,
}

pub fn run(addrs: Vec<String>, api_key: Option<String>) -> io::Result<()> {
    let nodes: Arc<Mutex<Vec<NodeStat>>> = Arc::new(Mutex::new(
        addrs
            .iter()
            .map(|a| NodeStat {
                addr: a.clone(),
                ..Default::default()
            })
            .collect(),
    ));

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let poller = {
        let nodes = Arc::clone(&nodes);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut prev: HashMap<String, (u64, Instant)> = HashMap::new();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let rows: Vec<NodeStat> =
                    addrs.iter().flat_map(|addr| poll(addr, api_key.as_deref(), &mut prev)).collect();
                *nodes.lock().unwrap() = rows;
                std::thread::sleep(Duration::from_millis(1000));
            }
        })
    };

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut term = Terminal::new(ratatui::backend::CrosstermBackend::new(io::stdout()))?;
    let start = Instant::now();
    let mut fleet_hist: VecDeque<u64> = VecDeque::with_capacity(HIST);
    let mut frame = 0usize;

    let res = loop {
        if event::poll(Duration::from_millis(200)).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.kind == KeyEventKind::Press && matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break Ok(());
                }
            }
        }
        frame = frame.wrapping_add(1);
        let snapshot: Vec<NodeStat> = nodes.lock().unwrap().clone();
        let total_tps: u64 = snapshot.iter().map(|n| n.tps).sum();
        if frame.is_multiple_of(5) {
            if fleet_hist.len() >= HIST {
                fleet_hist.pop_front();
            }
            fleet_hist.push_back(total_tps);
        }
        if let Err(e) = term.draw(|f| draw(f, &snapshot, &fleet_hist, total_tps, start, frame)) {
            break Err(e);
        }
    };

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = poller.join();
    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen)?;
    term.show_cursor()?;
    res
}

fn draw(f: &mut Frame, nodes: &[NodeStat], hist: &VecDeque<u64>, total_tps: u64, start: Instant, frame: usize) {
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(6),
        Constraint::Length(5),
        Constraint::Length(1),
    ])
    .split(f.area());

    let up = start.elapsed().as_secs();
    let healthy = nodes.iter().filter(|n| n.reachable || n.unauthorized).count();
    let unauthorized = nodes.iter().filter(|n| n.unauthorized).count();
    let spin = ["◐", "◓", "◑", "◒"][(frame / 2) % 4];
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!(" {spin} superfluid fleet "), Style::new().fg(Color::Black).bg(ACCENT).bold()),
            Span::raw(format!(
                "  {healthy}/{} nodes up{} · Σ {total_tps} tok/s · up {:02}:{:02}:{:02}",
                nodes.len(),
                if unauthorized > 0 { format!(" ({unauthorized} need --api-key)") } else { String::new() },
                up / 3600,
                (up % 3600) / 60,
                up % 60
            )),
        ])),
        rows[0],
    );

    let header = Row::new(["NODE", "STATE", "KV POOL", "TOK/S", "LANES", "QUEUE"].map(|h| {
        Cell::from(h).style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))
    }));
    let body: Vec<Row> = nodes
        .iter()
        .map(|n| {
            let state = if n.reachable {
                Cell::from("● up").style(Style::new().fg(Color::Green))
            } else if n.unauthorized {
                Cell::from("● 401").style(Style::new().fg(Color::Yellow))
            } else {
                Cell::from("● down").style(Style::new().fg(Color::Red))
            };
            let pool = if n.pool_total > 0 {
                let r = (n.pool_used as f64 / n.pool_total as f64).min(1.0);
                let filled = (r * 14.0) as usize;
                let bar = format!("{}{}", "█".repeat(filled), "░".repeat(14 - filled));
                Cell::from(Line::from(vec![
                    Span::styled(bar, Style::new().fg(util_color(r))),
                    // A pool sized for a long window is barely touched by short requests: in use
                    // is not empty.
                    Span::raw(match (r * 100.0) as u64 {
                        0 if n.pool_used > 0 => " <1%".to_string(),
                        pct => format!(" {pct}%"),
                    }),
                ]))
            } else {
                Cell::from(Span::styled("— n/a", Style::new().dim()))
            };
            let node_cell = if n.host.is_empty() {
                Cell::from(n.addr.clone())
            } else {
                Cell::from(Line::from(vec![
                    Span::styled(n.host.clone(), Style::new().add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  {}", n.addr), Style::new().dim()),
                ]))
            };
            Row::new(vec![
                node_cell,
                state,
                pool,
                Cell::from(format!("{}", n.tps)).style(Style::new().fg(ACCENT)),
                Cell::from(format!("{}", n.lanes)),
                Cell::from(format!("{}", n.queue)).style(if n.queue > 0 {
                    Style::new().fg(Color::Yellow)
                } else {
                    Style::new().dim()
                }),
            ])
        })
        .collect();
    let widths = [
        Constraint::Percentage(40),
        Constraint::Length(8),
        Constraint::Percentage(24),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(7),
    ];
    f.render_widget(
        Table::new(body, widths).header(header).block(bordered("nodes")).column_spacing(1),
        rows[1],
    );

    let data: Vec<u64> = hist.iter().copied().collect();
    let peak = data.iter().copied().max().unwrap_or(0);
    f.render_widget(
        Sparkline::default()
            .block(bordered(&format!("fleet decode tok/s  (peak {peak})")))
            .data(&data)
            .style(Style::new().fg(ACCENT)),
        rows[2],
    );

    f.render_widget(
        Paragraph::new(Span::styled(" q quit · polling /metrics @ 1 Hz ", Style::new().dim()))
            .alignment(Alignment::Center),
        rows[3],
    );
}

fn util_color(r: f64) -> Color {
    if r > 0.85 {
        Color::Red
    } else if r > 0.6 {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn bordered(title: &str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Rgb(70, 70, 80)))
        .title(Span::styled(format!(" {title} "), Style::new().fg(ACCENT).bold()))
}

/// The rows one address stands for: a server's own, or every node of a fleet head.
fn poll(addr: &str, api_key: Option<&str>, prev: &mut HashMap<String, (u64, Instant)>) -> Vec<NodeStat> {
    let m = match scrape(addr, api_key) {
        Ok(m) => m,
        Err(e) => {
            return vec![NodeStat {
                addr: addr.to_string(),
                unauthorized: matches!(e, Scrape::Unauthorized),
                ..Default::default()
            }]
        }
    };
    let now = Instant::now();
    let fleet = fleet_nodes(&m);
    if !fleet.is_empty() || m.keys().any(|k| k.starts_with("superfluid_fleet_head_info{")) {
        return fleet
            .into_iter()
            .map(|(labels, name, node_addr)| {
                let at = |series: &str| get(&m, &format!("{series}{labels}"));
                NodeStat {
                    tps: rate(prev, format!("{addr} {labels}"), at("superfluid_fleet_node_decode_tokens_total"), now),
                    addr: node_addr,
                    host: name,
                    reachable: at("superfluid_fleet_node_up") == 1,
                    unauthorized: false,
                    lanes: at("superfluid_fleet_node_lanes_active"),
                    pool_used: at("superfluid_fleet_node_pool_blocks_used"),
                    pool_total: at("superfluid_fleet_node_pool_blocks_total"),
                    queue: at("superfluid_fleet_node_queue_depth"),
                }
            })
            .collect();
    }
    vec![NodeStat {
        tps: rate(prev, addr.to_string(), get(&m, "superfluid_decode_tokens_total"), now),
        addr: addr.to_string(),
        host: host_of(&m).unwrap_or_default(),
        reachable: true,
        unauthorized: false,
        lanes: get(&m, "superfluid_lanes_active"),
        pool_used: get(&m, "superfluid_pool_blocks_used"),
        pool_total: get(&m, "superfluid_pool_blocks_total"),
        queue: ["interactive_chat", "inline_completion", "foreground_agent", "background_agent"]
            .iter()
            .map(|c| get(&m, &format!("superfluid_queue_depth{{class=\"{c}\"}}")))
            .sum(),
    }]
}

/// Tokens per second since the last poll of the same row; 0 on its first.
fn rate(prev: &mut HashMap<String, (u64, Instant)>, key: String, total: u64, now: Instant) -> u64 {
    match prev.insert(key, (total, now)) {
        Some((was, t)) => (total.saturating_sub(was) as f64 / (now - t).as_secs_f64().max(1e-3)) as u64,
        None => 0,
    }
}

/// A fleet head's nodes, in the order it lists them: the label set each series carries for the
/// node, its name and its address.
fn fleet_nodes(m: &HashMap<String, f64>) -> Vec<(String, String, String)> {
    let mut nodes: Vec<(String, String, String)> = m
        .keys()
        .filter_map(|k| k.strip_prefix("superfluid_fleet_node_up"))
        .filter(|labels| labels.starts_with('{'))
        .map(|labels| (labels.to_string(), label(labels, "node"), label(labels, "addr")))
        .collect();
    nodes.sort_by(|a, b| a.1.cmp(&b.1));
    nodes
}

/// A label's value in a `{k="v",...}` set; escapes are left as they are.
fn label(labels: &str, key: &str) -> String {
    labels
        .split(&format!("{key}=\""))
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .unwrap_or_default()
        .to_string()
}

enum Scrape {
    Unauthorized,
    Down,
}

fn scrape(addr: &str, api_key: Option<&str>) -> Result<HashMap<String, f64>, Scrape> {
    let url = format!("http://{addr}/metrics");
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(2)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let mut req = agent.get(&url);
    if let Some(key) = api_key {
        req = req.header("authorization", &format!("Bearer {key}"));
    }
    let mut resp = req.call().map_err(|_| Scrape::Down)?;
    match resp.status().as_u16() {
        200 => {}
        401 => return Err(Scrape::Unauthorized),
        _ => return Err(Scrape::Down),
    }
    let body = resp.body_mut().read_to_string().map_err(|_| Scrape::Down)?;
    Ok(parse_prometheus(&body))
}

fn parse_prometheus(text: &str) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(sp) = line.rfind(' ') {
            let (k, v) = line.split_at(sp);
            if let Ok(val) = v.trim().parse::<f64>() {
                m.insert(k.trim().to_string(), val);
            }
        }
    }
    m
}

fn get(m: &HashMap<String, f64>, key: &str) -> u64 {
    m.get(key).copied().unwrap_or(0.0) as u64
}

fn host_of(m: &HashMap<String, f64>) -> Option<String> {
    for k in m.keys() {
        if let Some(rest) = k.strip_prefix("superfluid_node_info{") {
            if let Some(h) = rest.split("host=\"").nth(1) {
                return h.split('"').next().map(|s| s.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn keyed_metrics_server(key: &'static str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            for sock in listener.incoming() {
                let Ok(mut sock) = sock else { break };
                let mut buf = [0u8; 4096];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_lowercase();
                let want = format!("authorization: bearer {key}");
                let resp = if req.contains(&want) {
                    let body = "superfluid_decode_tokens_total 7\n";
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                } else {
                    "HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
                };
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        addr
    }

    /// Serves what `body` holds now on every request, as a fleet head's `/metrics`.
    fn head_metrics_server(body: Arc<Mutex<String>>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            for sock in listener.incoming() {
                let Ok(mut sock) = sock else { break };
                let _ = sock.read(&mut [0u8; 4096]);
                let body = body.lock().unwrap().clone();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        addr
    }

    fn node(name: &str, addr: &str, up: bool, decode_tokens: u64) -> crate::fleet_manager::NodeStatus {
        crate::fleet_manager::NodeStatus {
            name: name.to_string(),
            addr: addr.to_string(),
            up,
            lanes: 8,
            lanes_active: 2,
            queue_depth: 1,
            kv_blocks_used: 25,
            kv_blocks_total: 100,
            decode_tokens,
        }
    }

    #[test]
    fn a_fleet_heads_address_stands_for_each_of_its_nodes() {
        let render = |nodes: &[crate::fleet_manager::NodeStatus]| {
            crate::fleet_openai::fleet_metrics(nodes, "head", "qwen")
        };
        let body = Arc::new(Mutex::new(render(&[node("m3-ultra", "10.0.0.57", true, 100)])));
        let head = head_metrics_server(Arc::clone(&body));
        let mut prev = HashMap::new();

        let rows = poll(&head, None, &mut prev);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!((r.host.as_str(), r.addr.as_str(), r.reachable), ("m3-ultra", "10.0.0.57", true));
        assert_eq!((r.lanes, r.queue, r.pool_used, r.pool_total, r.tps), (2, 1, 25, 100, 0));

        // A node joins and the first one generates: the rows follow without a restart.
        std::thread::sleep(Duration::from_millis(50));
        *body.lock().unwrap() =
            render(&[node("m3-ultra", "10.0.0.57", true, 200), node("spark-gb10", "10.0.0.22", false, 0)]);
        let rows = poll(&head, None, &mut prev);
        let names: Vec<_> = rows.iter().map(|r| (r.host.as_str(), r.reachable)).collect();
        assert_eq!(names, [("m3-ultra", true), ("spark-gb10", false)]);
        assert!(rows[0].tps > 0, "100 tokens since the last poll");
    }

    #[test]
    fn a_head_with_no_nodes_yet_shows_no_rows_not_itself() {
        let head = head_metrics_server(Arc::new(Mutex::new(crate::fleet_openai::fleet_metrics(&[], "head", "qwen"))));
        assert!(poll(&head, None, &mut HashMap::new()).is_empty());
    }

    #[test]
    fn scrape_sends_the_key_and_tells_401_from_down() {
        let addr = keyed_metrics_server("top-secret");
        assert!(matches!(scrape(&addr, None), Err(Scrape::Unauthorized)), "no key is 401, not down");
        assert!(matches!(scrape(&addr, Some("wrong")), Err(Scrape::Unauthorized)));
        let m = scrape(&addr, Some("top-secret")).ok().expect("the key opens /metrics");
        assert_eq!(get(&m, "superfluid_decode_tokens_total"), 7);
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
        assert!(matches!(scrape(&free, Some("top-secret")), Err(Scrape::Down)));
    }
}
