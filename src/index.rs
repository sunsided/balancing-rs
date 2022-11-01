use std::cell::{Ref, RefCell, RefMut};
use std::collections::hash_map::Values;
use std::collections::{BinaryHeap, HashMap};
use std::iter::Map;
use std::num::NonZeroUsize;

pub type ShardId = NonZeroUsize;

const SHARDID_ONE: ShardId = unsafe { NonZeroUsize::new_unchecked(1) };

#[derive(Debug)]
pub struct Index {
    pub index_id: usize,
    pub num_vectors: usize,
    pub vector_length: usize,
    shards: HashMap<ShardId, RefCell<IndexAssignment>>,
    highest_shard_id: ShardId,
}

impl Index {
    pub fn new(index_id: usize, num_items: usize, vector_length: usize) -> Self {
        let shard_id = Self::root_shard_id();
        let first_shard = IndexAssignment {
            index_id,
            shard_id,
            num_vectors: num_items,
            vector_length,
        };
        let mut shards = HashMap::new();
        shards.insert(shard_id, RefCell::new(first_shard));

        Self {
            index_id,
            num_vectors: num_items,
            vector_length,
            shards,
            highest_shard_id: shard_id,
        }
    }

    pub fn new_from_shards(index_id: usize, num_items: &[usize], vector_length: usize) -> Self {
        // If no shards are specified, start with the root shard of zero items.
        if num_items.is_empty() {
            Self::new(index_id, 0, vector_length);
        }

        let mut shard_id = Self::root_shard_id();
        let first_shard = IndexAssignment {
            index_id,
            shard_id,
            num_vectors: num_items[0],
            vector_length,
        };
        let mut shards = HashMap::new();
        shards.insert(shard_id, RefCell::new(first_shard));

        for i in 1..num_items.len() {
            shard_id = shard_id.checked_add(1).unwrap();
            let next_shard = IndexAssignment {
                index_id,
                shard_id,
                num_vectors: num_items[i],
                vector_length,
            };
            shards.insert(shard_id, RefCell::new(next_shard));
        }

        Self {
            index_id,
            num_vectors: num_items.iter().sum(),
            vector_length,
            shards,
            highest_shard_id: shard_id,
        }
    }

    pub fn weight(&self) -> usize {
        self.num_vectors * self.vector_length
    }

    pub fn create_empty_shard(&mut self) -> ShardId {
        let shard_id = self.new_shard_id();
        let assignment = IndexAssignment {
            index_id: self.index_id,
            shard_id,
            num_vectors: 0,
            vector_length: self.vector_length,
        };
        self.shards.insert(shard_id, RefCell::new(assignment));
        shard_id
    }

    pub fn shard(&self, shard_id: ShardId) -> Result<Ref<IndexAssignment>, GetShardError> {
        if let Some(shard) = self.shards.get(&shard_id) {
            return Ok(shard.borrow());
        }

        return Err(GetShardError::ShardNotFound { shard_id });
    }

    pub fn get_shard_mut(
        &self,
        shard_id: ShardId,
    ) -> Result<RefMut<IndexAssignment>, GetShardError> {
        if let Some(shard) = self.shards.get(&shard_id) {
            return Ok(shard.borrow_mut());
        }

        return Err(GetShardError::ShardNotFound { shard_id });
    }

    pub fn move_data(
        &self,
        source_shard_id: ShardId,
        target_shard_id: ShardId,
        amount: usize,
    ) -> Result<(Ref<IndexAssignment>, Ref<IndexAssignment>), AssignmentError> {
        if !self.shards.contains_key(&source_shard_id) {
            return Err(AssignmentError::ShardNotFound {
                shard_id: source_shard_id,
            });
        }

        if !self.shards.contains_key(&target_shard_id) {
            return Err(AssignmentError::ShardNotFound {
                shard_id: target_shard_id,
            });
        }

        let source_shard = self.shards.get(&source_shard_id).unwrap();
        let target_shard = self.shards.get(&target_shard_id).unwrap();

        {
            let mut source_shard = self.shards.get(&source_shard_id).unwrap().borrow_mut();
            let mut target_shard = self.shards.get(&target_shard_id).unwrap().borrow_mut();

            if source_shard.num_vectors < amount {
                return Err(AssignmentError::SourceShardTooSmall);
            }

            source_shard.num_vectors -= amount;
            target_shard.num_vectors += amount;
        }

        Ok((source_shard.borrow(), target_shard.borrow()))
    }

    pub fn shard_ids(&self) -> Vec<ShardId> {
        self.shards
            .keys()
            .cloned()
            .collect::<BinaryHeap<ShardId>>()
            .into_sorted_vec()
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn len(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    fn root_shard_id() -> ShardId {
        SHARDID_ONE
    }

    fn new_shard_id(&mut self) -> ShardId {
        self.highest_shard_id = self.highest_shard_id.checked_add(1).unwrap();
        self.highest_shard_id
    }
}

#[derive(thiserror::Error, Debug)]
pub enum GetShardError {
    #[error("The shard does not exist")]
    ShardNotFound { shard_id: ShardId },
}

#[derive(thiserror::Error, Debug)]
pub enum AssignmentError {
    #[error("The shard does not exist")]
    ShardNotFound { shard_id: ShardId },
    #[error("The source shard does not contain enough items")]
    SourceShardTooSmall,
}

#[derive(Debug, Clone)]
pub struct IndexAssignment {
    pub index_id: usize,
    pub shard_id: ShardId,
    pub num_vectors: usize,
    pub vector_length: usize,
}

impl IndexAssignment {
    pub fn weight(&self) -> usize {
        self.num_vectors * self.vector_length
    }

    pub fn len(&self) -> usize {
        self.num_vectors * self.vector_length
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

impl<'a> IntoIterator for &'a Index {
    type Item = Ref<'a, IndexAssignment>;
    type IntoIter = Map<
        Values<'a, ShardId, RefCell<IndexAssignment>>,
        fn(&RefCell<IndexAssignment>) -> Ref<IndexAssignment>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.shards.values().map(|rc| rc.borrow())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let index = Index::new(0, 100, 512);
        assert_eq!(index.len(), 100);
        assert_eq!(index.weight(), 51200);
        assert_eq!(index.num_shards(), 1);
        assert_eq!(
            index.shard(Index::root_shard_id()).unwrap().num_vectors,
            100
        );
    }

    #[test]
    fn new_many_works() {
        let index = Index::new_from_shards(0, &[50, 75], 512);
        let ids = index.shard_ids();
        assert_eq!(index.num_shards(), 2);
        assert_eq!(ids.len(), 2);
        assert_eq!(index.shard(ids[0]).unwrap().num_vectors, 50);
        assert_eq!(index.shard(ids[1]).unwrap().num_vectors, 75);
    }

    #[test]
    fn shard_assignment_works() {
        let mut index = Index::new(0, 100, 512);
        let old_shard_id = Index::root_shard_id();
        let new_shard_id = index.create_empty_shard();
        index.move_data(old_shard_id, new_shard_id, 100).unwrap();
        assert_eq!(index.shard(old_shard_id).unwrap().num_vectors, 0);
        assert_eq!(index.shard(new_shard_id).unwrap().num_vectors, 100);
    }
}
