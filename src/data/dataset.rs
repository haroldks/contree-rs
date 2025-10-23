use crate::data::{DataPoint, Feature};
use std::ops::{Index, IndexMut};

#[derive(Default)]
pub struct Dataset {
    features: Vec<Feature>,
    num_labels: usize,
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            features: vec![],
            num_labels: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.features[0].len()
    }

    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
    pub fn insert(&mut self, data_point: DataPoint, feature_index: usize) {
        if data_point.tid() == 0 {
            self.features.push(Feature::new(feature_index))
        }
        self.features[feature_index].insert(data_point)
    }

    pub fn set_num_label(&mut self, value: usize) {
        self.num_labels = value;
    }

    pub fn sort_features(&mut self) {
        for column in self.features.iter_mut() {
            column.sort();
        }
    }

    pub fn compute_unique_feature_values(&mut self) {
        let size = self.count();
        let mut idx = vec![0; size];
        for column in &mut self.features {
            (0..size).for_each(|i| idx[i] = i);
            idx.sort_unstable_by(|&idx1, &idx2| column[idx1].cmp(&column[idx2]));

            let mut cur_unique = 0;
            let mut prev: Option<f64> = None;
            for &index in &idx {
                let el = &mut column[index];
                if let Some(prev_val) = prev {
                    if (el.value - prev_val).abs() >= f64::EPSILON {
                        // TODO : The use a larger epsilon and not the absolute value
                        cur_unique += 1;
                    }
                }

                el.unique_value_idx = cur_unique;
                prev = Some(el.value);
            }
        }
    }
}

impl Index<usize> for Dataset {
    type Output = Feature;

    fn index(&self, index: usize) -> &Self::Output {
        &self.features[index]
    }
}

impl IndexMut<usize> for Dataset {
    fn index_mut(&mut self, index: usize) -> &mut Feature {
        &mut self.features[index]
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = &'a Feature;
    type IntoIter = std::slice::Iter<'a, Feature>;

    fn into_iter(self) -> Self::IntoIter {
        self.features.iter()
    }
}
