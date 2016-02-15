

As DHTs are optimized for reduced latency, increases to comorbidity between failures in adjacent nodes in the overlay begins to damage the robustness guarantees that were originally expected from DHT-like systems.

To mitigate potential data loss,a naive solution of randomly distributing backups across geographic space stands out. While this is an incredibly viable technique when focused on robustness, it means when we fall back to it due to node failures, we lose all benefits of the latency-optimized overlays.

To preform a compromise with robustness and latency I will explore a sampling of the techniques available for more efficient record backup
