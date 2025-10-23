use super::{DataFormat, DataReaderError};
use crate::bitsets::BitCollection;
use crate::data::{DataPoint, Dataset};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct DataReader {
    format: DataFormat,
    has_headers: bool,
    comment_char: Option<char>,
    label_column: Option<usize>,
}

impl Default for DataReader {
    fn default() -> Self {
        Self {
            format: DataFormat::Space,
            has_headers: false,
            comment_char: Some('#'),
            label_column: Some(0),
        }
    }
}

impl DataReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_format(mut self, format: DataFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    pub fn with_comment_char(mut self, comment_char: Option<char>) -> Self {
        self.comment_char = comment_char;
        self
    }

    pub fn with_label_column(mut self, label_column: Option<usize>) -> Self {
        self.label_column = label_column;
        self
    }

    pub fn auto_detect_format(mut self, path: &Path) -> Self {
        self.format = DataFormat::from_extension(path);
        self
    }

    pub fn read_file(&self, path: &Path) -> Result<Dataset, DataReaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let delimiter = self.format.delimiter();
        let mut row_idx = 0;

        let mut dataset = Dataset::new();
        let mut num_labels = HashSet::<usize>::new();

        for (i, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(comment) = self.comment_char {
                if line.starts_with(comment) {
                    continue;
                }
            }

            if i == 0 && self.has_headers {
                continue;
            }

            let tokens: Vec<&str> = if delimiter == ' ' {
                line.split_whitespace().collect()
            } else {
                line.split(delimiter)
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect()
            };
            if tokens.is_empty() {
                continue;
            }

            let label_col = self.label_column.unwrap_or(usize::MAX);
            if label_col >= tokens.len() {
                return Err(DataReaderError::Format(format!(
                    "Label column {} out of bounds at line {}",
                    label_col, i
                )));
            }

            let label: usize = tokens[label_col].parse::<usize>().map_err(|_| {
                DataReaderError::Parse(format!(
                    "Invalid label '{}' at line {} (column {})",
                    tokens[label_col], i, label_col
                ))
            })?;
            num_labels.insert(label);

            for (col_idx, &tok) in tokens.iter().enumerate() {
                if Some(col_idx) == self.label_column {
                    continue;
                }

                let effective_col = if col_idx > self.label_column.unwrap_or(usize::MAX) {
                    col_idx - 1
                } else {
                    col_idx
                };

                let v = tok.parse::<f64>().map_err(|_| {
                    DataReaderError::Parse(format!(
                        "Invalid float '{}' at line {} (column {})",
                        tok,
                        i + 1,
                        col_idx + 1
                    ))
                })?;

                dataset.insert(DataPoint::new(row_idx, v, label as f64), effective_col);
            }
            row_idx += 1;
        }

        dataset.set_num_label(num_labels.len());
        dataset.compute_unique_feature_values();
        Ok(dataset)
    }
}
