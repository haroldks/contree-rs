use crate::bitsets::{BitCollection, Bitset};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

// TODO: use private fields and updater
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub struct Entry {
    pub feature: usize,
    pub split: f64,
    pub age: usize,
    pub error: usize,
    pub label: usize,
    
    pub ub: usize,

    pub is_leaf: bool,
    pub is_optimal: bool,

    pub depth: usize,
    pub left: usize,
    pub right: usize,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            feature: usize::MAX,
            split: f64::INFINITY,
            age: 0,
            error: usize::MAX,
            label: usize::MAX,
            ub: 0,
            is_leaf: false,
            is_optimal: false,
            depth: 0,
            left: 0,
            right: 0,
        }
    }
}

pub struct Cache {
    depth: usize,
    num_samples: usize,
    arena: Vec<Entry>,
    map: Vec<Vec<FxHashMap<Bitset, usize>>>,
    root_index: usize,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl Cache {
    pub fn new(depth: usize, num_samples: usize) -> Self {
        Self {
            depth,
            num_samples,
            arena: Vec::new(),
            map: vec![vec![FxHashMap::default(); num_samples]; depth],
            root_index: 0,
        }
    }

    pub fn root(&self) -> Option<&Entry> {
        self.arena.get(self.root_index)
    }

    pub fn root_mut(&mut self) -> Option<&mut Entry> {
        self.arena.get_mut(self.root_index)
    }

    pub fn init(&mut self) -> usize {
        debug_assert!(self.arena.len() == 0, "Cache must me empty to init");
        self.arena.push(Entry::default());
        self.root_index
    }

    pub fn insert(&mut self, bitset: &Bitset, depth: usize) -> (bool, usize) {
        coz::scope!("insert in cache");
        let mut cache = &mut self.map[depth][bitset.count()];
        if cache.contains_key(&bitset) {
            (false, *cache.get(&bitset).unwrap())
        } else {
            let index = self.arena.len();
            self.arena.push(Entry::default());
            cache.insert(bitset.clone(), index);
            (true, index)
        }
    }

    pub fn get(&self, index: usize) -> Option<&Entry> {
        self.arena.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Entry> {
        self.arena.get_mut(index)
    }

    pub fn find(&self, bitset: &Bitset, depth: usize) -> Option<&usize> {
        debug_assert!(depth <= self.depth, "Depth out of range");
        debug_assert!(bitset.count() <= self.num_samples, "Bitset inconsistent");
        let cache = &self.map[depth][bitset.count()];
        cache.get(bitset)
    }
    
    pub fn len(&self) -> usize {
        self.arena.len()
    }
}
