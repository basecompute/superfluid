//! `superfluid serve` terminal UI — an htop/nvtop-style live monitor.

#![cfg(feature = "tui")]

use std::collections::VecDeque;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseEventKind,
};
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::crossterm::{execute, ExecutableCommand};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::symbols::Marker;
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Chart, Dataset, Gauge, GraphType, Paragraph,
};
use ratatui::{Frame, Terminal};
use unicode_width::UnicodeWidthChar;

use crate::scheduler::SchedStats;
use crate::telemetry::{push_ring_line, LogLines, LogRing};
use crate::Daemon;

static SAVED_STDERR: AtomicI32 = AtomicI32::new(-1);
static PIPE_W: AtomicI32 = AtomicI32::new(-1);
static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);
static TUI_THREAD: Mutex<Option<ThreadId>> = Mutex::new(None);
static PANIC_HOOK: Once = Once::new();
static DRAIN_REQUESTED: AtomicU64 = AtomicU64::new(0);
static DRAIN_SEEN: AtomicU64 = AtomicU64::new(0);
const DRAIN_MARK: &[u8] = b"\x00superfluid-stderr-drain:";
const MAX_LINE_BYTES: u64 = 16 * 1024;

pub struct StderrCapture {
    saved: OwnedFd,
    pipe_w: Arc<OwnedFd>,
}

impl StderrCapture {
    pub fn worker_stderr(&self) -> Arc<OwnedFd> {
        Arc::clone(&self.pipe_w)
    }

    pub fn drain(&self, timeout: std::time::Duration) {
        use std::io::Write;
        let seq = DRAIN_REQUESTED.fetch_add(1, Ordering::SeqCst) + 1;
        let mark = format!("\x00superfluid-stderr-drain:{seq}\n");
        // SAFETY: a borrowed, non-owning handle on the write end for the
        // duration of one write; the OwnedFd keeps it open.
        let mut w = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(self.pipe_w.as_raw_fd()) });
        if w.write_all(mark.as_bytes()).is_err() {
            return;
        }
        let deadline = Instant::now() + timeout;
        while DRAIN_SEEN.load(Ordering::SeqCst) < seq && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

pub fn capture_stderr(ring: LogRing) -> io::Result<StderrCapture> {
    use std::io::Write;
    // SAFETY: fd syscalls on descriptors this process owns; every fd is
    // wrapped in an OwnedFd as soon as it exists, so a failure path closes
    // what it opened.
    unsafe {
        let saved = libc::fcntl(2, libc::F_DUPFD_CLOEXEC, 3);
        if saved < 0 {
            return Err(io::Error::last_os_error());
        }
        let saved = OwnedFd::from_raw_fd(saved);
        let mut fds = [0i32; 2];
        if libc::pipe(fds.as_mut_ptr()) < 0 {
            return Err(io::Error::last_os_error());
        }
        let (pipe_r, pipe_w) = (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1]));
        libc::fcntl(pipe_r.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
        libc::fcntl(pipe_w.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
        let mirror = libc::fcntl(saved.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3);
        if mirror < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut mirror = std::fs::File::from_raw_fd(mirror);
        let mirror_always = libc::isatty(saved.as_raw_fd()) == 0;
        let mut reader = std::fs::File::from(pipe_r);
        std::thread::Builder::new()
            .name("superfluid-stderr-ring".into())
            .spawn(move || {
                use std::io::Read;
                let mut emit = |raw: &[u8]| {
                    let (raw, seen) = match raw.windows(DRAIN_MARK.len()).position(|w| w == DRAIN_MARK) {
                        Some(at) => {
                            let seq = std::str::from_utf8(&raw[at + DRAIN_MARK.len()..])
                                .ok()
                                .and_then(|t| t.trim_end().parse::<u64>().ok());
                            (&raw[..at], seq)
                        }
                        None => (raw, None),
                    };
                    for piece in raw.chunks(MAX_LINE_BYTES as usize) {
                        if mirror_always || !TUI_ACTIVE.load(Ordering::SeqCst) {
                            let _ = mirror.write_all(piece);
                        }
                        push_ring_line(&ring, &String::from_utf8_lossy(piece));
                    }
                    if let Some(seq) = seen {
                        DRAIN_SEEN.fetch_max(seq, Ordering::SeqCst);
                    }
                };
                let mut chunk = vec![0u8; MAX_LINE_BYTES as usize];
                let mut carry: Vec<u8> = Vec::new();
                loop {
                    let n = match reader.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    carry.extend_from_slice(&chunk[..n]);
                    while let Some(pos) = carry.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = carry.drain(..=pos).collect();
                        emit(&line);
                    }
                    while carry.len() as u64 >= MAX_LINE_BYTES {
                        let piece: Vec<u8> = carry.drain(..MAX_LINE_BYTES as usize).collect();
                        emit(&piece);
                    }
                }
                if !carry.is_empty() {
                    emit(&carry);
                }
            })?;
        SAVED_STDERR.store(saved.as_raw_fd(), Ordering::SeqCst);
        PIPE_W.store(pipe_w.as_raw_fd(), Ordering::SeqCst);
        PANIC_HOOK.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                let on_tui_thread = TUI_ACTIVE.load(Ordering::SeqCst)
                    && TUI_THREAD.lock().ok().and_then(|g| *g) == Some(std::thread::current().id());
                if on_tui_thread {
                    restore_terminal();
                }
                prev(info);
            }));
        });
        Ok(StderrCapture {
            saved,
            pipe_w: Arc::new(pipe_w),
        })
    }
}

impl Drop for StderrCapture {
    fn drop(&mut self) {
        if TUI_ACTIVE.load(Ordering::SeqCst) {
            restore_terminal();
        }
        let _ = SAVED_STDERR.compare_exchange(
            self.saved.as_raw_fd(),
            -1,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        let _ = PIPE_W.compare_exchange(self.pipe_w.as_raw_fd(), -1, Ordering::SeqCst, Ordering::SeqCst);
    }
}

fn restore_terminal() {
    TUI_ACTIVE.store(false, Ordering::SeqCst);
    if let Ok(mut g) = TUI_THREAD.lock() {
        *g = None;
    }
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
    let _ = io::stdout().execute(ratatui::crossterm::cursor::Show);
    let saved = SAVED_STDERR.load(Ordering::SeqCst);
    if saved >= 0 {
        // SAFETY: `saved` is a descriptor the live capture owns.
        unsafe {
            libc::dup2(saved, 2);
        }
    }
}

pub struct NodeInfo {
    pub identity: String,
    pub model: String,
    pub engine: String,
    pub max_lanes: u32,
}

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const HIST: usize = 160;
const ACCENT: Color = Color::Cyan;

pub fn run(
    daemon: Arc<Daemon>,
    ring: LogRing,
    node: NodeInfo,
    capture: Option<&StderrCapture>,
) -> io::Result<()> {
    struct RestoreGuard;
    impl Drop for RestoreGuard {
        fn drop(&mut self) {
            restore_terminal();
        }
    }
    let guard = RestoreGuard;
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    io::stdout().execute(EnableMouseCapture)?;
    let mut term = Terminal::new(ratatui::backend::CrosstermBackend::new(io::stdout()))?;
    if let Some(c) = capture {
        if let Ok(mut g) = TUI_THREAD.lock() {
            *g = Some(std::thread::current().id());
        }
        TUI_ACTIVE.store(true, Ordering::SeqCst);
        // SAFETY: dup2 of a descriptor the capture owns onto fd 2.
        unsafe {
            libc::dup2(c.pipe_w.as_raw_fd(), 2);
        }
    }

    let res = event_loop(&mut term, daemon, ring, node);

    drop(guard);
    res
}

fn event_loop<B: ratatui::backend::Backend>(
    term: &mut Terminal<B>,
    daemon: Arc<Daemon>,
    ring: LogRing,
    node: NodeInfo,
) -> io::Result<()> {
    let stats = daemon.sched_stats();
    let start = Instant::now();
    let mut tput: VecDeque<u64> = VecDeque::with_capacity(HIST);
    let mut pref: VecDeque<u64> = VecDeque::with_capacity(HIST);
    let mut kvh: VecDeque<u64> = VecDeque::with_capacity(HIST);
    let mut samples: VecDeque<(Instant, u64)> = VecDeque::with_capacity(8);
    samples.push_back((Instant::now(), stats.decode_tokens.load(Ordering::Relaxed)));
    let mut prev_t = Instant::now();
    let mut last_work_ticks = stats.work_ticks.load(Ordering::Relaxed);
    let mut frame = 0usize;
    let mut paused = false;
    let mut log = LogView::default();

    'ui: loop {
        if event::poll(Duration::from_millis(125))? {
            // Take everything queued before redrawing, so a burst of wheel
            // ticks or mouse moves costs one frame rather than one each.
            loop {
                match event::read()? {
                    Event::Mouse(m) => match m.kind {
                        MouseEventKind::ScrollUp => log.up(3),
                        MouseEventKind::ScrollDown => log.down(3),
                        _ => {}
                    },
                    Event::Key(k) if k.kind == KeyEventKind::Press => {
                        if k.code == KeyCode::Char('c')
                            && k.modifiers.contains(event::KeyModifiers::CONTROL)
                        {
                            break 'ui;
                        }
                        match k.code {
                            KeyCode::Char('q') | KeyCode::Esc => break 'ui,
                            KeyCode::Char('p') => paused = !paused,
                            KeyCode::Up => log.up(1),
                            KeyCode::Down => log.down(1),
                            KeyCode::PageUp => log.up(log.page_step()),
                            KeyCode::PageDown => log.down(log.page_step()),
                            KeyCode::Home | KeyCode::Char('g') => log.top(),
                            KeyCode::End | KeyCode::Char('G') => log.follow(),
                            _ => {}
                        }
                    }
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }

        let now = Instant::now();
        let dt = (now - prev_t).as_secs_f64();
        if !paused && dt >= 0.20 {
            let decode = stats.decode_tokens.load(Ordering::Relaxed);
            samples.push_back((now, decode));
            while samples.len() > 2 && now.duration_since(samples[1].0) >= Duration::from_secs(1) {
                samples.pop_front();
            }
            let (t_old, d_old) = samples[0];
            let win = now.duration_since(t_old).as_secs_f64().max(1e-3);
            push(&mut tput, ((decode.saturating_sub(d_old)) as f64 / win) as u64);
            let work_ticks = stats.work_ticks.load(Ordering::Relaxed);
            let fresh = work_ticks != last_work_ticks;
            last_work_ticks = work_ticks;
            let resident = stats.lanes_active.load(Ordering::Relaxed) > 0;
            push(&mut pref, if resident || fresh { stats.prefill_rate_live.load(Ordering::Relaxed) } else { 0 });
            push(&mut kvh, stats.pool_blocks_used.load(Ordering::Relaxed));
            prev_t = now;
        }
        frame = frame.wrapping_add(1);

        term.draw(|f| {
            draw(f, &node, &stats, start, &tput, &pref, &kvh, frame, paused, &ring, &mut log)
        })?;
    }
    Ok(())
}

fn push(q: &mut VecDeque<u64>, v: u64) {
    if q.len() >= HIST {
        q.pop_front();
    }
    q.push_back(v);
}

#[allow(clippy::too_many_arguments)]
fn draw(
    f: &mut Frame,
    node: &NodeInfo,
    s: &SchedStats,
    start: Instant,
    tput: &VecDeque<u64>,
    pref: &VecDeque<u64>,
    kvh: &VecDeque<u64>,
    frame: usize,
    paused: bool,
    ring: &LogRing,
    log: &mut LogView,
) {
    let g = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(7),
        Constraint::Length(14),
        Constraint::Length(5),
        Constraint::Min(6),
        Constraint::Length(1),
    ])
    .split(f.area());

    let spin = SPINNER[(frame / 2) % SPINNER.len()];
    let up = start.elapsed().as_secs();
    let header = Line::from(vec![
        Span::styled(format!(" {spin} superfluid "), Style::new().fg(Color::Black).bg(ACCENT).bold()),
        Span::styled(format!(" {} ", node.identity), Style::new().fg(ACCENT).bold()),
        Span::raw(format!("· {} · engine {} · up {:02}:{:02}:{:02}", node.model, node.engine, up / 3600, (up % 3600) / 60, up % 60)),
        Span::styled(if paused { "  [PAUSED]" } else { "" }, Style::new().fg(Color::Yellow).bold()),
    ]);
    f.render_widget(Paragraph::new(vec![header, in_tick_line(s, rows[0].width)]), rows[0]);

    let node_cols = Layout::horizontal([Constraint::Percentage(48), Constraint::Percentage(52)]).split(rows[1]);
    let gauges = Layout::vertical([Constraint::Length(3), Constraint::Length(3)]).split(node_cols[0]);
    let lanes = g(&s.lanes_active);
    let maxl = node.max_lanes as u64;
    let lr = if maxl > 0 { (lanes as f64 / maxl as f64).min(1.0) } else { 0.0 };
    f.render_widget(
        Gauge::default()
            .block(bordered("lane occupancy"))
            .gauge_style(Style::new().fg(util_color(lr)).bg(Color::Rgb(28, 28, 32)))
            .ratio(lr)
            .label(format!("{lanes}/{maxl} lanes · {}% · {} free", (lr * 100.0) as u64, maxl.saturating_sub(lanes))),
        gauges[0],
    );
    let used = g(&s.pool_blocks_used);
    let total = g(&s.pool_blocks_total);
    if total > 0 {
        let pr = (used as f64 / total as f64).min(1.0);
        f.render_widget(
            Gauge::default()
                .block(bordered("KV pool"))
                .gauge_style(Style::new().fg(util_color(pr)).bg(Color::Rgb(28, 28, 32)))
                .ratio(pr)
                .label(format!("{used}/{total} blk · {}% · {} free", (pr * 100.0) as u64, total - used)),
            gauges[1],
        );
    } else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("blocks in use  ", Style::new().dim()),
                    Span::styled(format!("{used}"), Style::new().fg(ACCENT).bold()),
                ]),
                Line::from(Span::styled("capacity n/a (engine)", Style::new().dim().italic())),
            ])
            .block(bordered("KV pool")),
            gauges[1],
        );
    }
    let ttft = s.ttft_last_ms.load(Ordering::Relaxed);
    let warm = warm_pct(s);
    let (sa, sp) = (g(&s.spec_accepted), g(&s.spec_proposed));
    let spec = (100 * sa).checked_div(sp).unwrap_or(0);
    let dec = g(&s.decode_rate_live);
    let pre = g(&s.prefill_rate_live);
    let idle = lanes == 0;
    let rate = |v: u64, col: Color| -> (String, Color) {
        if idle {
            (format!("{v:>6} tok/s  last"), Color::DarkGray)
        } else {
            (format!("{v:>6} tok/s"), col)
        }
    };
    let (dec_s, dec_c) = rate(dec, ACCENT);
    let (pre_s, pre_c) = rate(pre, Color::Magenta);
    let stat_lines = vec![
        kv("decode", dec_s, dec_c),
        kv("prefill", pre_s, pre_c),
        kv("ttft last", format!("{ttft:>6} ms"), ttft_color(ttft)),
        kv("warm reuse", format!("{warm:>5}%"), Color::Green),
        kv("spec accept", format!("{spec:>5}%"), Color::Blue),
    ];
    f.render_widget(Paragraph::new(stat_lines).block(bordered("throughput")), node_cols[1]);

    let mid = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)]).split(rows[2]);
    let left = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).split(mid[0]);
    let right = Layout::vertical([Constraint::Length(6), Constraint::Min(4)]).split(mid[1]);
    rate_chart(f, left[0], "decode · delivered tok/s · 1 s window", tput, ACCENT, None);
    rate_chart(f, left[1], "prefill · engine tok/s · per tick", pref, Color::Magenta, None);
    level_chart(f, right[1], "KV pool · blocks used", kvh, Color::Green);
    let classes = ["chat", "compl", "fg-agent", "bg-agent"];
    let qlines: Vec<Line> = classes
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let d = g(&s.queue_depth[i]);
            let bar = "▊".repeat((d.min(20)) as usize);
            Line::from(vec![
                Span::styled(format!("{name:>8} "), Style::new().dim()),
                Span::styled(bar, Style::new().fg(if d == 0 { Color::DarkGray } else { Color::Yellow })),
                Span::raw(format!(" {d}")),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(qlines).block(bordered("queue depth / QoS")), right[0]);

    let c = |name: &str, v: u64, col: Color| {
        vec![
            Span::styled(format!("{name} "), Style::new().dim()),
            Span::styled(format!("{v}"), Style::new().fg(col).bold()),
            Span::raw("   "),
        ]
    };
    let mut cline: Vec<Span> = Vec::new();
    cline.extend(c("ticks", g(&s.ticks), Color::White));
    cline.extend(c("parks✓", g(&s.parks_lossless), Color::Green));
    cline.extend(c("parks~", g(&s.parks_lossy), Color::Yellow));
    cline.extend(c("resumes", g(&s.resumes), Color::Green));
    cline.extend(c("preempt", g(&s.preemptions), Color::Magenta));
    let mut cline2: Vec<Span> = Vec::new();
    cline2.extend(c("evictions", g(&s.pressure_evictions), Color::Yellow));
    cline2.extend(c("respawns", g(&s.worker_respawns), Color::Red));
    cline2.extend(c("cold", g(&s.cold_admissions), Color::Blue));
    cline2.extend(c("starve-grants", g(&s.starvation_grants), Color::Magenta));
    cline2.extend(c("os-pressure", g(&s.os_pressure_events), Color::Red));
    let idle_now = lanes == 0;
    let mut cline3: Vec<Span> = Vec::new();
    cline3.extend(c(
        if idle_now { "last-tick(ms)" } else { "tick(ms)" },
        g(&s.tick_wall_ms_live),
        if idle_now { Color::DarkGray } else { Color::White },
    ));
    cline3.extend(c("grant(tok/lane)", g(&s.decode_grant_live), Color::Cyan));
    cline3.extend(c("prefill-budget(tok)", g(&s.prefill_budget_live), Color::Magenta));
    f.render_widget(
        Paragraph::new(vec![Line::from(cline), Line::from(cline2), Line::from(cline3)])
            .block(bordered("counters")),
        rows[3],
    );

    render_log(f, rows[4], ring, log);

    f.render_widget(
        Paragraph::new(Line::from(vec![Span::styled(
            " q quit · p pause · ↑↓/PgUp/PgDn/wheel scroll · Home top · End follow ",
            Style::new().dim(),
        )]))
        .alignment(Alignment::Center),
        rows[5],
    );
}

fn rate_chart(f: &mut Frame, area: Rect, title: &str, data: &VecDeque<u64>, col: Color, ymax: Option<u64>) {
    let n = data.len();
    let now = data.back().copied().unwrap_or(0);
    let peak = data.iter().copied().max().unwrap_or(0);
    let top = ymax.unwrap_or_else(|| nice_ceil(peak));
    let pts: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| ((HIST - n + i) as f64, (*v).min(top) as f64))
        .collect();
    let ds = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(col))
        .data(&pts);
    let mid = top / 2;
    let chart = Chart::new(vec![ds])
        .block(bordered(&format!("{title}  ·  now {now} · peak {peak}")))
        .x_axis(Axis::default().bounds([0.0, (HIST - 1) as f64]))
        .y_axis(
            Axis::default()
                .bounds([0.0, top as f64])
                .style(Style::new().dim())
                .labels([
                    Span::styled("0", Style::new().dim()),
                    Span::styled(format!("{mid}"), Style::new().dim()),
                    Span::styled(format!("{top}"), Style::new().dim()),
                ]),
        );
    f.render_widget(chart, area);
}

fn level_chart(f: &mut Frame, area: Rect, title: &str, data: &VecDeque<u64>, col: Color) {
    let n = data.len();
    let now = data.back().copied().unwrap_or(0);
    let hi = data.iter().copied().max().unwrap_or(0);
    let lo = data.iter().copied().min().unwrap_or(0);
    let (bottom, top) = level_band(lo, hi);
    let pts: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| ((HIST - n + i) as f64, (*v).clamp(bottom, top) as f64))
        .collect();
    let ds = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(col))
        .data(&pts);
    let mid = bottom + (top - bottom) / 2;
    let chart = Chart::new(vec![ds])
        .block(bordered(&format!("{title}  ·  now {now} · low {lo} · high {hi}")))
        .x_axis(Axis::default().bounds([0.0, (HIST - 1) as f64]))
        .y_axis(
            Axis::default()
                .bounds([bottom as f64, top as f64])
                .style(Style::new().dim())
                .labels([
                    Span::styled(format!("{bottom}"), Style::new().dim()),
                    Span::styled(format!("{mid}"), Style::new().dim()),
                    Span::styled(format!("{top}"), Style::new().dim()),
                ]),
        );
    f.render_widget(chart, area);
}

fn level_band(lo: u64, hi: u64) -> (u64, u64) {
    let span = hi.saturating_sub(lo).max(1);
    let mut step = 1u64;
    while step.saturating_mul(10) <= span {
        step = step.saturating_mul(10);
    }
    let step = step.max(10);
    let bottom = (lo / step) * step;
    let top = hi.div_ceil(step) * step;
    let top = if top == bottom { bottom + step } else { top };
    (bottom.min(lo), top.max(hi))
}

fn nice_ceil(v: u64) -> u64 {
    if v <= 10 {
        return 10;
    }
    let mut m = 1u64;
    while m.saturating_mul(10) <= v {
        m = m.saturating_mul(10);
    }
    for k in [1u64, 2, 5, 10] {
        if k.saturating_mul(m) >= v {
            return k.saturating_mul(m);
        }
    }
    10 * m
}

fn in_tick_line(s: &SchedStats, width: u16) -> Line<'static> {
    let Some(t) = s.tick_in_flight() else {
        return Line::from(Span::styled("   between ticks", Style::new().dim()));
    };
    let age_ms = crate::scheduler::uptime_ms().saturating_sub(t.since_ms);
    let age = if age_ms < 1000 || t.target_ms < 1000 {
        format!("{age_ms}ms")
    } else {
        format!("{:.1}s", age_ms as f64 / 1e3)
    };
    let target = (t.target_ms as f64).max(1.0);
    let col = if age_ms as f64 > 3.0 * target {
        Color::Red
    } else if age_ms as f64 > 1.25 * target {
        Color::Yellow
    } else {
        Color::Green
    };
    let lanes = |n: u64| format!("{n} lane{}", if n == 1 { "" } else { "s" });
    let mut segs: Vec<String> = Vec::new();
    if t.prefill_tokens > 0 {
        segs.push(format!("prefill {} tok/{}", t.prefill_tokens, lanes(t.prefill_lanes)));
    }
    if t.decode_lanes > 0 {
        segs.push(format!("decode {}", lanes(t.decode_lanes)));
    }
    if t.admits > 0 {
        segs.push(format!("admit {}", t.admits));
    }
    if t.retires > 0 {
        segs.push(format!("retire {}", t.retires));
    }
    if segs.is_empty() {
        segs.push("empty plan".into());
    }
    let prefix = format!("   in tick {age}: ");
    let budget = (width as usize).saturating_sub(prefix.chars().count());
    let mut text = String::new();
    let mut dropped = false;
    for (i, seg) in segs.iter().enumerate() {
        let piece = if i == 0 { seg.clone() } else { format!(" · {seg}") };
        let tail = if i + 1 < segs.len() { 2 } else { 0 };
        if text.chars().count() + piece.chars().count() + tail <= budget {
            text.push_str(&piece);
        } else {
            dropped = true;
            break;
        }
    }
    if dropped {
        text.push_str(" …");
    }
    Line::from(Span::styled(format!("{prefix}{text}"), Style::new().fg(col).bold()))
}

/// Where the log pane sits: `scroll` screen rows up from the newest one, so
/// 0 follows the tail. It counts rows rather than lines because long lines
/// wrap, and it is clamped on every draw so scrolling past the top never
/// builds up slack that Down would first have to work off.
#[derive(Default)]
struct LogView {
    scroll: usize,
    /// `LogLines::end` at the last draw. Rows that arrive while scrolled up
    /// are added to `scroll`, so the text being read holds still.
    seen_end: Option<u64>,
    /// Rows the pane showed at the last draw.
    page: usize,
}

impl LogView {
    fn up(&mut self, rows: usize) {
        self.scroll = self.scroll.saturating_add(rows);
    }
    fn down(&mut self, rows: usize) {
        self.scroll = self.scroll.saturating_sub(rows);
    }
    fn top(&mut self) {
        self.scroll = usize::MAX;
    }
    fn follow(&mut self) {
        self.scroll = 0;
    }
    /// A page less one row, so the row at the edge stays in view.
    fn page_step(&self) -> usize {
        self.page.saturating_sub(1).max(1)
    }
}

fn render_log(f: &mut Frame, area: Rect, ring: &LogRing, view: &mut LogView) {
    let (w, h) = (area.width.saturating_sub(2) as usize, area.height.saturating_sub(2) as usize);
    let rows = {
        let log = ring.lock().unwrap_or_else(|p| p.into_inner());
        log_window(&log, w, h, view)
    };
    let title = if view.scroll == 0 {
        "logs (following)".to_string()
    } else {
        format!("logs (−{})", view.scroll)
    };
    f.render_widget(Paragraph::new(Text::from(rows)).block(bordered(&title)), area);
}

/// The `height` rows of `log`, wrapped to `width` columns, that `view`
/// points at, oldest first. Clamps `view.scroll` to the oldest row and
/// shifts it by whatever arrived since the last call.
fn log_window(log: &LogLines, width: usize, height: usize, view: &mut LogView) -> Vec<Line<'static>> {
    let end = log.end();
    if let Some(seen) = view.seen_end.filter(|_| view.scroll > 0) {
        let fresh = end.saturating_sub(seen).min(log.len() as u64) as usize;
        let rows: usize = log.iter().rev().take(fresh).map(|l| wrap_line(l, width).count()).sum();
        view.scroll = view.scroll.saturating_add(rows);
    }
    view.seen_end = Some(end);
    view.page = height;

    // Walk back from the newest line until the rows below the window's top
    // are covered; running out first means the window is at the top.
    let want = view.scroll.saturating_add(height);
    let (mut first, mut below) = (log.len(), 0usize);
    while first > 0 && below < want {
        first -= 1;
        below += wrap_line(&log[first], width).count();
    }
    view.scroll = view.scroll.min(below.saturating_sub(height));

    let bottom = below - view.scroll;
    let top = bottom.saturating_sub(height);
    log.range(first..)
        .flat_map(|l| {
            let style = log_style(l);
            wrap_line(l, width).map(move |row| Line::styled(row.to_string(), style))
        })
        .take(bottom)
        .skip(top)
        .collect()
}

/// `line` cut into rows at most `width` columns wide, between characters.
fn wrap_line(line: &str, width: usize) -> WrapLine<'_> {
    WrapLine { rest: line, width: width.max(1) }
}

struct WrapLine<'a> {
    rest: &'a str,
    width: usize,
}

impl<'a> Iterator for WrapLine<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        // No character is wider than its UTF-8 encoding is long.
        if self.rest.len() <= self.width {
            return Some(std::mem::take(&mut self.rest));
        }
        let (mut cols, mut cut) = (0, self.rest.len());
        for (i, c) in self.rest.char_indices() {
            let w = c.width().unwrap_or(0);
            if w > 0 && cols > 0 && cols + w > self.width {
                cut = i;
                break;
            }
            cols += w;
        }
        let (row, rest) = self.rest.split_at(cut);
        self.rest = rest;
        Some(row)
    }
}

fn log_style(l: &str) -> Style {
    if l.contains("ERROR") {
        Style::new().fg(Color::Red)
    } else if l.contains("WARN") {
        Style::new().fg(Color::Yellow)
    } else if l.contains("DEBUG") || l.contains("TRACE") {
        Style::new().fg(Color::DarkGray).add_modifier(Modifier::DIM)
    } else {
        Style::new().fg(Color::Gray)
    }
}

fn bordered(title: &str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Rgb(70, 70, 80)))
        .title(Span::styled(format!(" {title} "), Style::new().fg(ACCENT).bold()))
}

fn kv(k: &str, v: String, col: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{k:>11}  "), Style::new().dim()),
        Span::styled(v, Style::new().fg(col).bold()),
    ])
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
fn ttft_color(ms: u64) -> Color {
    if ms > 800 {
        Color::Red
    } else if ms > 300 {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn warm_pct(s: &SchedStats) -> u64 {
    let warm = s.warm_prefix_tokens.load(Ordering::Relaxed);
    let cold = s.prefill_tokens.load(Ordering::Relaxed);
    let tot = warm + cold;
    (100 * warm).checked_div(tot).unwrap_or(0)
}

#[cfg(test)]
mod chart_tests {
    use super::level_band;

    #[test]
    fn level_band_frames_the_window_with_round_headroom() {
        assert_eq!(level_band(9369, 9800), (9300, 9800));
        assert_eq!(level_band(9369, 9369), (9360, 9370));
        assert_eq!(level_band(0, 0), (0, 10));
        assert_eq!(level_band(3, 7), (0, 10));
        assert_eq!(level_band(1200, 31000), (0, 40000));
        for (lo, hi) in [(0, 1), (5, 5), (999, 1001), (12345, 67890)] {
            let (b, t) = level_band(lo, hi);
            assert!(b <= lo && t >= hi && t > b, "{lo}..{hi} -> {b}..{t}");
        }
    }
}

#[cfg(test)]
mod log_pane_tests {
    use super::*;
    use crate::telemetry::new_log_ring;

    fn ring_of(lines: impl IntoIterator<Item = String>) -> LogRing {
        let ring = new_log_ring();
        for l in lines {
            push_ring_line(&ring, &l);
        }
        ring
    }

    fn numbered(n: usize) -> LogRing {
        ring_of((0..n).map(|i| format!("l{i}")))
    }

    fn shown(ring: &LogRing, width: usize, height: usize, view: &mut LogView) -> Vec<String> {
        let log = ring.lock().unwrap();
        log_window(&log, width, height, view)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn wrap_line_cuts_on_display_columns() {
        fn rows(s: &str, w: usize) -> Vec<&str> {
            wrap_line(s, w).collect()
        }
        assert_eq!(rows("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        assert_eq!(rows("abcd", 4), ["abcd"]);
        assert_eq!(rows("日本語ab", 4), ["日本", "語ab"]);
        assert_eq!(rows("abcde\u{301}f", 5), ["abcde\u{301}", "f"], "a combining mark stays on its letter");
        assert_eq!(rows("ab", 0), ["a", "b"]);
        assert!(rows("", 4).is_empty());
    }

    #[test]
    fn following_shows_the_newest_rows_when_the_tail_wraps() {
        let ring = ring_of(["l1", "l2", "l3", "0123456789abcdefghij"].map(String::from));
        let mut view = LogView::default();
        assert_eq!(shown(&ring, 10, 3, &mut view), ["l3", "0123456789", "abcdefghij"]);
        view.up(1);
        assert_eq!(shown(&ring, 10, 3, &mut view), ["l2", "l3", "0123456789"]);
    }

    #[test]
    fn scrolling_past_the_top_leaves_no_slack() {
        let ring = numbered(10);
        let mut view = LogView::default();
        view.up(100);
        assert_eq!(shown(&ring, 10, 4, &mut view), ["l0", "l1", "l2", "l3"]);
        assert_eq!(view.scroll, 6);
        view.down(1);
        assert_eq!(shown(&ring, 10, 4, &mut view), ["l1", "l2", "l3", "l4"]);

        view.top();
        assert_eq!(shown(&ring, 10, 4, &mut view), ["l0", "l1", "l2", "l3"]);
        view.down(view.page_step());
        assert_eq!(shown(&ring, 10, 4, &mut view), ["l3", "l4", "l5", "l6"]);
        view.follow();
        assert_eq!(shown(&ring, 10, 4, &mut view), ["l6", "l7", "l8", "l9"]);
    }

    #[test]
    fn fewer_lines_than_rows_sit_at_the_top_and_cannot_scroll() {
        let ring = numbered(2);
        let mut view = LogView::default();
        view.up(5);
        assert_eq!(shown(&ring, 10, 4, &mut view), ["l0", "l1"]);
        assert_eq!(view.scroll, 0);
        assert!(shown(&new_log_ring(), 10, 4, &mut view).is_empty());
    }

    #[test]
    fn new_lines_do_not_move_a_scrolled_up_view() {
        let ring = numbered(10);
        let mut view = LogView::default();
        view.up(2);
        assert_eq!(shown(&ring, 10, 3, &mut view), ["l5", "l6", "l7"]);
        push_ring_line(&ring, "new\nwraps-onto-two");
        assert_eq!(shown(&ring, 10, 3, &mut view), ["l5", "l6", "l7"]);
        assert_eq!(view.scroll, 5);

        let mut tail = LogView::default();
        assert_eq!(shown(&ring, 10, 3, &mut tail), ["new", "wraps-onto", "-two"]);
    }

    #[test]
    fn a_scrolled_up_view_holds_while_the_ring_evicts() {
        let ring = numbered(2100);
        let mut view = LogView::default();
        view.up(5);
        let before = shown(&ring, 20, 3, &mut view);
        assert_eq!(before, ["l2092", "l2093", "l2094"]);
        for i in 0..7 {
            push_ring_line(&ring, &format!("n{i}"));
        }
        assert_eq!(shown(&ring, 20, 3, &mut view), before);
    }

    #[test]
    fn keys_before_the_first_draw_do_not_count_the_backlog_as_new() {
        let ring = numbered(10);
        let mut view = LogView::default();
        view.up(1);
        assert_eq!(shown(&ring, 10, 3, &mut view), ["l6", "l7", "l8"]);
    }

    #[test]
    fn the_pane_renders_wrapped_rows_inside_its_border() {
        use ratatui::backend::TestBackend;
        let ring = ring_of(["older".to_string(), format!("WARN {}", "x".repeat(20))]);
        let mut term = Terminal::new(TestBackend::new(22, 5)).unwrap();
        let mut view = LogView::default();
        term.draw(|f| render_log(f, f.area(), &ring, &mut view)).unwrap();
        let buf = term.backend().buffer();
        let row = |y: u16| (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>();
        assert!(row(0).contains("logs (following)"), "{:?}", row(0));
        assert_eq!(row(1), "│older               │");
        assert_eq!(row(2), "│WARN xxxxxxxxxxxxxxx│");
        assert_eq!(row(3), "│xxxxx               │");
        assert_eq!(buf[(1, 2)].fg, Color::Yellow);
        assert_eq!(buf[(1, 3)].fg, Color::Yellow, "a wrapped row keeps its line's colour");
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    use std::io::Write;
    use std::time::Duration;

    #[test]
    fn worker_end_lines_reach_the_ring_and_fd2_is_untouched() {
        let ring = crate::telemetry::new_log_ring();
        // SAFETY: fstat of our own fd 2 into a zeroed buffer.
        let before = unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            assert_eq!(libc::fstat(2, &mut st), 0);
            (st.st_dev, st.st_ino)
        };
        let cap = capture_stderr(ring.clone()).expect("capture");
        let w = cap.worker_stderr();
        let mut f = std::fs::File::from(w.try_clone().expect("clone"));
        f.write_all(b"[capture-test] one\n").unwrap();
        f.write_all(b"[capture-test] \xff\xfe bad\n").unwrap();
        let long = vec![b'x'; (MAX_LINE_BYTES as usize) * 2 + 7];
        f.write_all(&long).unwrap();
        f.write_all(b"\n[capture-test] two\n[capture-test] three\n[capture-test] four\n").unwrap();
        cap.drain(Duration::from_secs(5));
        {
            let lines = ring.lock().unwrap();
            for want in ["[capture-test] two", "[capture-test] three", "[capture-test] four"] {
                assert!(lines.iter().any(|l| l == want), "after drain, missing {want}: {:?}", *lines);
            }
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let n = ring.lock().unwrap().iter().filter(|l| l.contains("[capture-test]")).count();
            if n >= 3 {
                break;
            }
            assert!(Instant::now() < deadline, "ring never saw the lines: {:?}", ring.lock().unwrap());
            std::thread::sleep(Duration::from_millis(10));
        }
        // SAFETY: as above.
        let after = unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            assert_eq!(libc::fstat(2, &mut st), 0);
            (st.st_dev, st.st_ino)
        };
        assert_eq!(before, after, "installing the capture must not move fd 2");
        let lines: Vec<String> = ring.lock().unwrap().iter().cloned().collect();
        assert!(lines.iter().any(|l| l == "[capture-test] one"), "{lines:?}");
        assert!(lines.iter().any(|l| l == "[capture-test] two"), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("[capture-test] ") && l.ends_with(" bad")), "{lines:?}");
        let xs: usize = lines.iter().filter(|l| l.bytes().all(|b| b == b'x')).map(|l| l.len()).sum();
        assert_eq!(xs, (MAX_LINE_BYTES as usize) * 2 + 7, "long run published in pieces: {lines:?}");
        assert!(lines.iter().filter(|l| l.bytes().all(|b| b == b'x')).all(|l| l.len() <= MAX_LINE_BYTES as usize));
        drop(f);
        drop(w);
        drop(cap);
    }
}
