//! The weights-free tokenizer contract.

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;

    fn token_bytes(&self, token: u32) -> Vec<u8>;

    fn vocab_size(&self) -> u32;

    fn special_tokens(&self) -> Vec<(String, u32)>;

    fn bos_token(&self) -> Option<u32>;

    fn eos_token(&self) -> u32;

    fn chat_template_jinja(&self) -> String;

    fn chat_template_named(&self, _name: &str) -> Option<String> {
        None
    }

    fn encode_plain(&self, _text: &str) -> Option<Vec<u32>> {
        None
    }

    fn encode_pieces(&self, _pieces: &[(&str, bool)]) -> Option<Vec<u32>> {
        None
    }
}

pub struct Markers {
    strings: std::collections::HashSet<Vec<u8>>,
    lengths: Vec<usize>,
    first: [bool; 256],
    bos: Option<u32>,
    bos_text: Option<String>,
    auto_bos: bool,
}

fn marker_shaped(v: &str) -> bool {
    let b = v.as_bytes();
    if b.len() >= 3 && b[0] == b'<' && b[b.len() - 1] == b'>' {
        (b.len() >= 4 && (b[1] == b'|' || b[b.len() - 2] == b'|'))
            || matches!(
                v,
                "<s>" | "</s>"
                    | "<unk>"
                    | "<pad>"
                    | "<bos>"
                    | "<eos>"
                    | "<sop>"
                    | "<eop>"
                    | "<think>"
                    | "</think>"
                    | "<tool_call>"
                    | "</tool_call>"
                    | "<tool_response>"
                    | "</tool_response>"
                    | "<arg_key>"
                    | "</arg_key>"
                    | "<arg_value>"
                    | "</arg_value>"
            )
    } else {
        matches!(v, "[gMASK]" | "[sMASK]" | "[MASK]" | "/nothink")
    }
}

impl Markers {
    pub fn of(tok: &dyn Tokenizer) -> Markers {
        let specials = tok.special_tokens();
        let bos = tok.bos_token();
        let bos_text = bos.and_then(|b| specials.iter().find(|(s, id)| *id == b && !s.is_empty()).map(|(s, _)| s.clone()));
        let mut templates = tok.chat_template_jinja();
        templates.push_str(&tok.chat_template_named("tool_use").unwrap_or_default());
        let mut first = [false; 256];
        let mut strings = std::collections::HashSet::new();
        for (text, id) in &specials {
            if text.trim().is_empty() {
                continue;
            }
            let control = tok.token_bytes(*id).is_empty();
            if control || marker_shaped(text) || templates.contains(text.as_str()) {
                first[text.as_bytes()[0] as usize] = true;
                strings.insert(text.as_bytes().to_vec());
            }
        }
        let mut lengths: Vec<usize> = strings.iter().map(Vec::len).collect();
        lengths.sort_unstable();
        lengths.dedup();
        let auto_bos = bos.is_some() && tok.encode("a").first().copied() == bos;
        Markers { strings, lengths, first, bos, bos_text, auto_bos }
    }

    fn occurrences(&self, text: &str) -> Vec<(usize, usize)> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        for (i, b) in bytes.iter().enumerate() {
            if !self.first[*b as usize] {
                continue;
            }
            for &len in &self.lengths {
                if i + len > bytes.len() {
                    break;
                }
                if self.strings.contains(&bytes[i..i + len]) {
                    out.push((i, i + len));
                }
            }
        }
        out
    }

    fn push(&self, tok: &dyn Tokenizer, text: &str, dedup: bool, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut toks = tok.encode(text);
        if self.auto_bos && toks.first().copied() == self.bos {
            let front = out.is_empty();
            let doubled = dedup
                && self.bos_text.as_deref().is_some_and(|b| text.starts_with(b))
                && toks.get(1).copied() == self.bos;
            if !front || doubled {
                toks.remove(0);
            }
        }
        out.extend(toks);
    }

    fn push_plain(&self, tok: &dyn Tokenizer, text: &str, out: &mut Vec<u32>) {
        let mut at = 0;
        let mut hits = self.occurrences(text);
        hits.sort_by_key(|&(start, end)| (start, std::cmp::Reverse(end)));
        for (start, end) in hits {
            if start < at {
                continue;
            }
            self.push(tok, &text[at..start], false, out);
            let marker = &text[start..end];
            let chars = marker.chars().count();
            match marker.char_indices().nth(chars / 2).filter(|_| chars > 1) {
                Some((mid, _)) => {
                    self.push_plain(tok, &marker[..mid], out);
                    self.push_plain(tok, &marker[mid..], out);
                }
                None => self.push(tok, marker, false, out),
            }
            at = end;
        }
        self.push(tok, &text[at..], false, out);
    }

    pub fn encode_plain(&self, tok: &dyn Tokenizer, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.push_plain(tok, text, &mut out);
        out
    }

    pub fn encode_pieces(&self, tok: &dyn Tokenizer, pieces: &[(&str, bool)]) -> Vec<u32> {
        let whole: String = pieces.iter().map(|(text, _)| *text).collect();
        let hits = self.occurrences(&whole);
        let mut touched = vec![false; pieces.len()];
        let mut at = 0;
        for (i, (text, content)) in pieces.iter().enumerate() {
            let (start, end) = (at, at + text.len());
            at = end;
            touched[i] = *content && start < end && hits.iter().any(|&(s, e)| s < end && e > start);
        }
        let mut out = Vec::new();
        if !touched.contains(&true) {
            self.push(tok, &whole, true, &mut out);
            return out;
        }
        let mut run = String::new();
        for (i, (text, _)) in pieces.iter().enumerate() {
            if touched[i] {
                self.push(tok, &run, true, &mut out);
                run.clear();
                self.push_plain(tok, text, &mut out);
            } else {
                run.push_str(text);
            }
        }
        self.push(tok, &run, true, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Toy {
        specials: Vec<(String, u32)>,
        bos: Option<u32>,
        text_like: Vec<u32>,
        template: String,
    }

    impl Tokenizer for Toy {
        fn encode(&self, text: &str) -> Vec<u32> {
            let mut out: Vec<u32> = self.bos.into_iter().collect();
            let bytes = text.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                let hit = self.specials.iter().filter(|(s, _)| bytes[i..].starts_with(s.as_bytes())).max_by_key(|(s, _)| s.len());
                match hit {
                    Some((s, id)) => {
                        out.push(*id);
                        i += s.len();
                    }
                    None => {
                        out.push(bytes[i] as u32);
                        i += 1;
                    }
                }
            }
            out
        }
        fn token_bytes(&self, token: u32) -> Vec<u8> {
            if token < 256 {
                return vec![token as u8];
            }
            let text = self.specials.iter().find(|(_, id)| *id == token).filter(|_| self.text_like.contains(&token));
            text.map(|(s, _)| s.as_bytes().to_vec()).unwrap_or_default()
        }
        fn vocab_size(&self) -> u32 {
            256 + self.specials.len() as u32
        }
        fn special_tokens(&self) -> Vec<(String, u32)> {
            self.specials.clone()
        }
        fn bos_token(&self) -> Option<u32> {
            self.bos
        }
        fn eos_token(&self) -> u32 {
            301
        }
        fn chat_template_jinja(&self) -> String {
            self.template.clone()
        }
    }

    const BOS: u32 = 300;
    const END: u32 = 301;
    const THINK: u32 = 302;
    const CLOSE: u32 = 303;

    const BLANK: u32 = 304;
    const TD: u32 = 305;
    const CALL: u32 = 306;

    fn toy(bos: bool) -> Toy {
        Toy {
            specials: vec![
                ("<s>".into(), BOS),
                ("<|end|>".into(), END),
                ("<think>".into(), THINK),
                ("</think>".into(), CLOSE),
                ("\n\n".into(), BLANK),
                ("<td>".into(), TD),
                ("<fn_call>".into(), CALL),
            ],
            bos: bos.then_some(BOS),
            text_like: vec![THINK, CLOSE, BLANK, TD, CALL],
            template: "{{ '<fn_call>' + call + '<|end|>' }}".into(),
        }
    }

    fn bytes(s: &str) -> Vec<u32> {
        s.bytes().map(u32::from).collect()
    }

    fn text_of(tok: &Toy, ids: &[u32]) -> String {
        ids.iter()
            .map(|&t| match tok.specials.iter().find(|(_, id)| *id == t) {
                Some((s, _)) => s.clone(),
                None => String::from_utf8(vec![t as u8]).unwrap(),
            })
            .collect()
    }

    #[test]
    fn content_without_a_marker_is_one_encode_of_the_whole_text() {
        for bos in [false, true] {
            let tok = toy(bos);
            let m = Markers::of(&tok);
            let pieces = [("user\n", false), ("hello\n", true), ("<|end|>\n", false), ("ok", true)];
            let whole: String = pieces.iter().map(|(t, _)| *t).collect();
            assert_eq!(m.encode_pieces(&tok, &pieces), tok.encode(&whole), "bos={bos}");
            assert_eq!(m.encode_plain(&tok, "plain words"), tok.encode("plain words"), "bos={bos}");
        }
    }

    #[test]
    fn a_marker_string_in_content_stays_words_and_one_in_framing_parses() {
        for bos in [false, true] {
            let tok = toy(bos);
            let m = Markers::of(&tok);
            let pieces = [("user\n", false), ("stop at <|end|> or <think>?", true), ("<|end|>", false)];
            let got = m.encode_pieces(&tok, &pieces);
            let lead: Vec<u32> = if bos { vec![BOS] } else { vec![] };
            let mut want = lead.clone();
            want.extend(bytes("user\nstop at <|end|> or <think>?"));
            want.push(END);
            assert_eq!(got, want, "bos={bos}: the quoted markers are bytes, the framing one is the id, one BOS at most");
            assert_eq!(got.iter().filter(|&&t| t == END).count(), 1);
            assert!(!got.contains(&THINK));
            let mut plain = lead;
            plain.extend(bytes("a <think> b"));
            assert_eq!(m.encode_plain(&tok, "a <think> b"), plain);
        }
    }

    #[test]
    fn added_tokens_that_are_text_stay_tokens_in_content() {
        for bos in [false, true] {
            let tok = toy(bos);
            let m = Markers::of(&tok);
            let text = "a\n\nb <td>cell";
            let whole = tok.encode(text);
            assert!(whole.contains(&BLANK) && whole.contains(&TD), "the vocabulary's own tokens: {whole:?}");
            assert_eq!(m.encode_plain(&tok, text), whole, "bos={bos}");
            let pieces = [("user\n", false), (text, true), ("<|end|>", false)];
            assert_eq!(m.encode_pieces(&tok, &pieces), tok.encode(&format!("user\n{text}<|end|>")), "bos={bos}");
            let got = m.encode_pieces(&tok, &[("<fn_call>", false), ("say <fn_call>", true)]);
            assert_eq!(got.iter().filter(|&&t| t == CALL).count(), 1, "bos={bos}: {got:?}");
            assert_eq!(text_of(&tok, &got).trim_start_matches("<s>"), "<fn_call>say <fn_call>");
        }
    }

    #[test]
    fn a_marker_reaching_into_content_from_framing_is_not_formed() {
        let tok = toy(false);
        let m = Markers::of(&tok);
        let got = m.encode_pieces(&tok, &[("x<|en", false), ("d|> tail", true)]);
        assert_eq!(got, bytes("x<|end|> tail"));
        let got = m.encode_plain(&tok, "</think>");
        assert_eq!(got, bytes("</think>"));
        assert_eq!(text_of(&tok, &got), "</think>");
    }

    #[test]
    fn the_automatic_bos_leads_once_whatever_the_template_renders() {
        let tok = toy(true);
        let m = Markers::of(&tok);
        let got = m.encode_pieces(&tok, &[("<s>user\n", false), ("hi", true)]);
        let mut want = vec![BOS];
        want.extend(bytes("user\nhi"));
        assert_eq!(got, want);
        let got = m.encode_pieces(&tok, &[("<s>user\n", false), ("<|end|>", true), ("\n", false)]);
        let mut want = vec![BOS];
        want.extend(bytes("user\n<|end|>\n"));
        assert_eq!(got, want);
        let got = m.encode_pieces(&tok, &[("user\n", false), ("hi", true)]);
        assert_eq!(got[0], BOS);
        assert_eq!(got.iter().filter(|&&t| t == BOS).count(), 1);
        let got = m.encode_pieces(&tok, &[("<s>", true), ("x", false)]);
        let mut want = vec![BOS];
        want.extend(bytes("<s>x"));
        assert_eq!(got, want);
        let tok = toy(false);
        let m = Markers::of(&tok);
        let got = m.encode_pieces(&tok, &[("<s>user\n", false), ("hi", true)]);
        let mut want = vec![BOS];
        want.extend(bytes("user\nhi"));
        assert_eq!(got, want);
    }
}
