
As we increase the scale and efficiency of DHT based content storage, many of the tenuous assumptions made by early DHTs begin to systematically fail. The largest of these assumptions is that any node can reasonably connect to any other given node in the network.
In extant systems this is most visible in the case of the NAT traversal problem which has been solved by re-centralization or application of super-nodes. This simply moves the assumption of "randomly-accessable" connectivity onto the super nodes or central NAT-traversal point, assuming that these points can effectively communicate with ALL of the nodes unable to connect.

This approach is a result of a design choice for DHTs that was reasonable until the networks approach the problem scale, that each node should be able and responsible for building a route and connection over the substrate network. As our scale and usage of this system grows, the reality that many computers on the earth would require DHT nodes that communicate over a variety of protocols and mechanisms need to be able to utilize an arbitrary number of bridge nodes to overcome the difficulties of the substrate network. In practice, the goal is for what was once an overlay network to become in practice a meta-network protocol, that would allow for the organization and distribution of all of human knowledge and much of it's communications.

As DHTs are optimized for reduced latency, increases to comorbidity between failures in adjacent nodes in the overlay begins to damage the robustness guarantees that were originally expected from DHT-like systems.

To mitigate potential data loss,a naive solution of randomly distributing backups across geographic space stands out. While this is an incredibly viable technique when focused on robustness, it means when we fall back to it due to node failures, we lose all benefits of the latency-optimized overlays.

To preform a compromise with robustness and latency I will explore a sampling of the techniques available for more efficient record backups.

I identify two types of robustness that must be considered in many areas:

- General failure robustness:
  - This is simply a syptom of random hardware failures and geographic failures in a well connected envoirment.
- Partition failure robustness:
  - This failure is caused by the random or geographic failure of nodes connecting a subgraph of the network to the rest of network.
  - This case ranges in magnitude from small groups of users being isolated by a gateway node's failure or



Popularity based strategic replication

 - Randomized distribution of replicas
  - Simply replicate popular files to more deterministic backup location, the  distribution of shortest distance from any given user to files will be biased low by popularity.
  - This approach only works effectively if no portions of the network are likely to be isolated by failure.
 - Strategic distribution of replicas
