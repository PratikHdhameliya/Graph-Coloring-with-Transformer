def test_folder(model, data_list, passes=5):
    """
    Returns the average unsatisfied% (avg_unsat)
    across all graphs in 'data_list'.
    """
    if len(data_list) == 0:
        return 0.0
    
    total_unsat = 0.0
    with torch.no_grad():
        for (n_nodes, edges) in data_list:
            if n_nodes < 1:
                continue
            # multiple inference passes => best solution
            best_probs, best_unsat_pct = multiple_inference_passes_parallel(
                model, n_nodes, edges, passes=passes
            )
            total_unsat += best_unsat_pct

    avg_unsat = total_unsat / len(data_list)
    return avg_unsat


def test_saved_model_on_4folders(
    checkpoint_path="checkpoints/best_model.pth",
    cc_folder="cc_graphs",
    geo_folder="geo_graphs",
    pwl_folder="pwl_graphs",
    gnm_folder="gnm_graphs",
    num_colors=3,
    embed_dim=32,
    n_layers=10,
    passes=5,
    device="cpu"
):
    """
    1) Load DistAwareColoringTransformer from `checkpoint_path`.
    2) Load .dimacs graphs from cc_folder, geo_folder, pwl_folder, gnm_folder.
    3) Evaluate average unsatisfied% on each folder with multiple_inference_passes_parallel.
    4) Print results.
    """

    # 1) Build the model structure & load checkpoint
    model = DistAwareColoringTransformer(
        embed_dim=embed_dim,
        num_colors=num_colors,
        n_layers=n_layers
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")

    # 2) Load each folder of .dimacs
    cc_data  = load_dimacs_graphs(cc_folder)
    geo_data = load_dimacs_graphs(geo_folder)
    pwl_data = load_dimacs_graphs(pwl_folder)
    gnm_data = load_dimacs_graphs(gnm_folder)

    print(f"Loaded {len(cc_data)} from '{cc_folder}', "
          f"{len(geo_data)} from '{geo_folder}', "
          f"{len(pwl_data)} from '{pwl_folder}', "
          f"{len(gnm_data)} from '{gnm_folder}'.")

    # 3) Evaluate average unsatisfied% on each folder
    avg_cc  = evaluate_folder(model, cc_data,  passes=passes)
    avg_geo = evaluate_folder(model, geo_data, passes=passes)
    avg_pwl = evaluate_folder(model, pwl_data, passes=passes)
    avg_gnm = evaluate_folder(model, gnm_data, passes=passes)

    # 4) Print
    print("\nAverage Unsatisfied% on each folder:")
    print(f"  Caveman (cc_graphs): {avg_cc:.2f}%")
    print(f"  Geometric (geo_graphs): {avg_geo:.2f}%")
    print(f"  Powerlaw (pwl_graphs): {avg_pwl:.2f}%")
    print(f"  Gnm (gnm_graphs): {avg_gnm:.2f}%")