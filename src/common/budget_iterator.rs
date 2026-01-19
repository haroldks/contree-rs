pub struct BudgetIterator {
    max_dis: usize,
    max_split: usize,
    current_sum: usize,
    current_dis: usize,
}

impl BudgetIterator {
    pub fn new(max_dis: usize, max_split: usize) -> Self {
        Self {
            max_dis,
            max_split,
            current_sum: 1,
            current_dis: 0,
        }
    }
}

impl Iterator for BudgetIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let max_total_sum = self.max_dis + self.max_split;

        while self.current_sum <= max_total_sum {
            // Constraint: split >= 1  =>  (current_sum - dis) >= 1  =>  dis <= current_sum - 1
            let max_dis_for_split_limit = self.current_sum.saturating_sub(1);

            // Calculate valid range for current_dis
            let min_dis_for_sum = if self.current_sum > self.max_split {
                self.current_sum - self.max_split
            } else {
                0
            };

            let max_dis_for_sum = max_dis_for_split_limit.min(self.max_dis);

            // Ensure current_dis is within the window
            if self.current_dis < min_dis_for_sum {
                self.current_dis = min_dis_for_sum;
            }

            if self.current_dis <= max_dis_for_sum {
                let dis = self.current_dis;
                let split = self.current_sum - dis;

                self.current_dis += 1;
                return Some((dis, split));
            } else {
                // Move to next diagonal
                self.current_sum += 1;
                self.current_dis = 0;
            }
        }
        None
    }
}
