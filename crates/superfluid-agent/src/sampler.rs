//! Host-side sampling.

pub fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_deterministic_on_ties() {
        assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.2]), 1);
        assert_eq!(argmax(&[0.5]), 0);
    }
}
