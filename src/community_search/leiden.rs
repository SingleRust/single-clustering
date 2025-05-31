use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use single_utilities::traits::{FloatOpsTS, ZeroVec};
use crate::moving::fast::FastLocalMoving;
use crate::moving::merging::LocalMerging;
use crate::network::grouping::{NetworkGrouping, VectorGrouping};
use crate::network::Network;

pub struct Leiden<T>
where
    T: FloatOpsTS,
{
    resolution: T,
    randomness: T,
    rng: ChaCha20Rng,
    local_moving: FastLocalMoving<T>,
    num_nodes_per_cluster_reduced_network: Vec<usize>,
}

impl<T> Leiden<T>
where
    T: FloatOpsTS + 'static,
{
    pub fn new(resolution: T, randomness: T, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_default();
        println!("WARNING!!!!! This implementation extremely highly unfinished and will be moved to a separate package in the future!");
        Leiden {
            resolution,
            randomness,
            rng: ChaCha20Rng::seed_from_u64(seed),
            local_moving: FastLocalMoving::new(resolution),
            num_nodes_per_cluster_reduced_network: Vec::new(),
        }
    }

    pub fn iterate(&mut self, network: &Network<T, T>, clustering: &mut VectorGrouping) -> bool {
        let mut update = self
            .local_moving
            .iterate(network, clustering, &mut self.rng);

        if clustering.node_count() == network.nodes() {
            return update;
        }

        let mut local_merging = LocalMerging::new(self.resolution, self.randomness);

        let subnetworks = network.create_subnetworks(clustering);

        let nodes_per_cluster = clustering.get_group_members();

        clustering.clear();

        self.num_nodes_per_cluster_reduced_network
            .zero_len(subnetworks.len());
        let mut cluster_counter = 0;

        for i in 0..subnetworks.len() {
            let sub_clustering = local_merging.run(&subnetworks[i], &mut self.rng);

            for j in 0..subnetworks[i].nodes() {
                clustering.set_group(
                    nodes_per_cluster[i][j],
                    cluster_counter + sub_clustering.get_group(j),
                );
            }

            cluster_counter += sub_clustering.group_count();
            self.num_nodes_per_cluster_reduced_network[i] = sub_clustering.group_count();
        }

        clustering.normalize_groups();

        let reduced_network = network.create_reduced_network(clustering);
        let mut clusters_reduced_network = vec![0; clustering.group_count()];

        let mut i = 0;
        for (j, num_nodes) in self
            .num_nodes_per_cluster_reduced_network
            .iter()
            .enumerate()
        {
            for cluster in clusters_reduced_network.iter_mut().skip(i).take(*num_nodes) {
                *cluster = j;
            }
            i += num_nodes;
        }

        let mut clustering_reduced_network =
            VectorGrouping::from_assignments(&clusters_reduced_network);
        update |= self.iterate(&reduced_network, &mut clustering_reduced_network);

        clustering.merge(&clustering_reduced_network);

        update
    }
}