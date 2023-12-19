import pandas as pd
import numpy as np
from deepchem.feat.graph_data import GraphData


def stack_graph_arrays(df):
    # Extract graph information
    tmp = pd.DataFrame(
        df["Graph"].to_list(),
        columns=["adj", "node_feats", "edge_feats", "adj_shape",
                 "node_feats_shape", "edge_feats_shape"],
    )

    # Recover original shapes
    tmp["adj"] = tmp.apply(lambda x: np.array(x["adj"]).reshape(
        [int(i) for i in x["adj_shape"]]), axis=1)
    tmp["node_feats"] = tmp.apply(
        lambda x: np.array(x["node_feats"]).reshape([int(i) for i in x["node_feats_shape"]]), axis=1
    )
    tmp["edge_feats"] = tmp.apply(
        lambda x: np.array(x["edge_feats"]).reshape([int(i) for i in x["edge_feats_shape"]]), axis=1
    )

    def build_graph_object(row):
        # Get dense edge index
        edge_idx = np.where(row["adj"])
        edge_index = np.stack([edge_idx[0], edge_idx[1]])

        # Get dense edge features
        edge_feat = row["edge_feats"][edge_idx[0], edge_idx[1], :]

        # Node features
        node_feats = row["node_feats"]

        return GraphData(node_feats, edge_index, edge_feat)

    return tmp.apply(lambda x: build_graph_object(x), axis=1).tolist()
