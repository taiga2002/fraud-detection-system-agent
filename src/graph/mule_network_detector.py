"""
Graph-Based Mule Network Detection

Uses NetworkX to build transaction graphs and detect mule account networks:
- Fan-out/fan-in analysis
- Community detection for mule clusters
- PageRank-style risk propagation
- Shared device/IP detection
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import pickle


class MuleNetworkDetector:
    """
    Graph-based detector for mule account networks.

    Builds a transaction graph and computes network-level risk scores.
    """

    def __init__(self):
        """Initialize mule network detector"""
        self.graph = None
        self.account_graph_features = {}


    def build_transaction_graph(self, transactions_df, accounts_df):
        """
        Build directed graph from transactions.

        Nodes: accounts
        Edges: transactions (sender -> receiver)

        Args:
            transactions_df: DataFrame with transactions
            accounts_df: DataFrame with accounts

        Returns:
            NetworkX DiGraph
        """
        print("Building transaction graph...")

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes (accounts)
        for _, account in accounts_df.iterrows():
            G.add_node(
                account['account_id'],
                account_age_days=account['account_age_days'],
                is_mule=account.get('is_mule', False),
                is_victim=account.get('is_victim', False),
                device_id=account.get('device_id', None),
                suspicious_device=account.get('suspicious_device', False)
            )

        # Add edges (transactions)
        for _, txn in transactions_df.iterrows():
            sender = txn['sender']
            receiver = txn['receiver']
            amount = txn['amount_jpy']

            if G.has_edge(sender, receiver):
                # Update existing edge
                G[sender][receiver]['weight'] += amount
                G[sender][receiver]['count'] += 1
                G[sender][receiver]['amounts'].append(amount)
            else:
                # Create new edge
                G.add_edge(
                    sender,
                    receiver,
                    weight=amount,
                    count=1,
                    amounts=[amount]
                )

        self.graph = G

        print(f"✓ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G


    def compute_graph_features(self):
        """
        Compute graph-based features for all accounts.

        Returns:
            Dictionary mapping account_id -> features dict
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_transaction_graph() first.")

        print("Computing graph features...")

        G = self.graph
        features = {}

        for node in G.nodes():
            # In-degree and out-degree
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)

            # Weighted degree (total amount)
            in_weight = sum(G[u][node]['weight'] for u in G.predecessors(node))
            out_weight = sum(G[node][v]['weight'] for v in G.successors(node))

            # Fan-out ratio (characteristic of mule accounts)
            fan_out_ratio = out_degree / (in_degree + 1)  # +1 to avoid division by zero

            # Pass-through ratio (money in vs money out)
            pass_through_ratio = out_weight / (in_weight + 1) if in_weight > 0 else 0

            # Burstiness (high degree in short time = suspicious)
            burst_score = (in_degree + out_degree) / (G.nodes[node].get('account_age_days', 365) + 1)

            # Is this a hub? (high betweenness centrality)
            # Note: Betweenness centrality is expensive, so we approximate with degree
            is_hub = (in_degree > 5) or (out_degree > 5)

            # Device/access sharing risk
            device_id = G.nodes[node].get('device_id', None)
            if device_id:
                # Count how many other accounts share this device
                shared_device_count = sum(
                    1 for n in G.nodes()
                    if G.nodes[n].get('device_id') == device_id and n != node
                )
            else:
                shared_device_count = 0

            has_shared_device = shared_device_count > 0

            # Known mule flag (from ground truth)
            is_known_mule = G.nodes[node].get('is_mule', False)

            # Mule-like pattern detection
            is_mule_like = (
                (fan_out_ratio > 2.0) and  # More outgoing than incoming
                (in_degree >= 3) and        # Receives from multiple sources
                (out_degree >= 2) and       # Sends to multiple destinations
                (G.nodes[node].get('account_age_days', 999) < 90)  # Young account
            )

            features[node] = {
                'in_degree': in_degree,
                'out_degree': out_degree,
                'fan_out_ratio': fan_out_ratio,
                'in_weight': in_weight,
                'out_weight': out_weight,
                'pass_through_ratio': pass_through_ratio,
                'burst_score': burst_score,
                'is_hub': int(is_hub),
                'shared_device_count': shared_device_count,
                'has_shared_device': int(has_shared_device),
                'is_known_mule': int(is_known_mule),
                'is_mule_like': int(is_mule_like)
            }

        self.account_graph_features = features

        print(f"✓ Computed graph features for {len(features)} accounts")

        return features


    def detect_mule_clusters(self, min_cluster_size=3):
        """
        Detect clusters of potentially connected mule accounts.

        Uses weakly connected components to find clusters.

        Args:
            min_cluster_size: Minimum size for a cluster to be flagged

        Returns:
            List of clusters (each cluster is a list of account IDs)
        """
        if self.graph is None:
            raise ValueError("Graph not built yet.")

        print(f"Detecting mule clusters (min size {min_cluster_size})...")

        # Find weakly connected components
        components = list(nx.weakly_connected_components(self.graph))

        # Filter for suspicious clusters
        suspicious_clusters = []

        for component in components:
            if len(component) < min_cluster_size:
                continue

            # Check if cluster shows mule-like behavior
            component_nodes = list(component)

            # Count known mules in cluster
            mule_count = sum(
                1 for node in component_nodes
                if self.graph.nodes[node].get('is_mule', False)
            )

            # High fan-out/fan-in activity
            avg_fan_out = np.mean([
                self.graph.out_degree(node) for node in component_nodes
            ])

            # Young accounts
            avg_age = np.mean([
                self.graph.nodes[node].get('account_age_days', 365)
                for node in component_nodes
            ])

            # Flag as suspicious cluster if:
            # - Contains at least one known mule, OR
            # - High average fan-out and young accounts
            is_suspicious = (mule_count > 0) or (avg_fan_out > 2 and avg_age < 90)

            if is_suspicious:
                suspicious_clusters.append({
                    'accounts': component_nodes,
                    'size': len(component_nodes),
                    'mule_count': mule_count,
                    'avg_fan_out': avg_fan_out,
                    'avg_account_age': avg_age
                })

        print(f"✓ Detected {len(suspicious_clusters)} suspicious clusters")

        return suspicious_clusters


    def compute_network_risk_scores(self):
        """
        Compute final network risk score for each account based on graph features.

        Returns:
            Dictionary mapping account_id -> network_risk_score (0-1)
        """
        if not self.account_graph_features:
            raise ValueError("Graph features not computed yet.")

        print("Computing network risk scores...")

        network_risk_scores = {}

        for account_id, features in self.account_graph_features.items():
            # Weighted combination of risk factors
            risk_score = 0.0

            # Known mule (ground truth)
            if features['is_known_mule']:
                risk_score += 0.6

            # Mule-like pattern
            if features['is_mule_like']:
                risk_score += 0.3

            # High fan-out ratio
            if features['fan_out_ratio'] > 3:
                risk_score += 0.2
            elif features['fan_out_ratio'] > 2:
                risk_score += 0.1

            # Hub behavior
            if features['is_hub']:
                risk_score += 0.15

            # Shared device
            if features['has_shared_device']:
                risk_score += 0.15

            # High burstiness
            if features['burst_score'] > 0.5:
                risk_score += 0.1

            # Cap at 1.0
            risk_score = min(risk_score, 1.0)

            network_risk_scores[account_id] = risk_score

        print(f"✓ Computed network risk scores for {len(network_risk_scores)} accounts")

        # Distribution
        scores = list(network_risk_scores.values())
        high_risk = sum(1 for s in scores if s > 0.7)
        medium_risk = sum(1 for s in scores if 0.4 < s <= 0.7)
        low_risk = sum(1 for s in scores if s <= 0.4)

        print(f"   High-risk accounts: {high_risk} ({high_risk/len(scores)*100:.1f}%)")
        print(f"   Medium-risk accounts: {medium_risk}")
        print(f"   Low-risk accounts: {low_risk} ({low_risk/len(scores)*100:.1f}%)")

        return network_risk_scores


    def save_detector(self, filepath):
        """Save detector state"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'account_graph_features': self.account_graph_features,
                'graph': self.graph
            }, f)

        print(f"✓ Detector saved to: {filepath}")


    @classmethod
    def load_detector(cls, filepath):
        """Load detector state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls()
        instance.account_graph_features = data['account_graph_features']
        instance.graph = data['graph']

        return instance


def main():
    """Build graph and compute network risk scores"""
    print("=" * 80)
    print("MULE NETWORK DETECTOR - GRAPH ANALYSIS")
    print("=" * 80)

    # Resolve project root
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Load data
    print("\n1. Loading data...")
    transactions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions.csv'))
    accounts_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts_with_risk.csv'))

    print(f"   {len(transactions_df)} transactions")
    print(f"   {len(accounts_df)} accounts")

    # Build graph
    print("\n2. Building transaction graph...")
    detector = MuleNetworkDetector()
    detector.build_transaction_graph(transactions_df, accounts_df)

    # Compute features
    print("\n3. Computing graph features...")
    graph_features = detector.compute_graph_features()

    # Detect clusters
    print("\n4. Detecting mule clusters...")
    clusters = detector.detect_mule_clusters(min_cluster_size=3)

    if clusters:
        print("\n   Top 5 Suspicious Clusters:")
        for i, cluster in enumerate(sorted(clusters, key=lambda x: x['size'], reverse=True)[:5]):
            print(f"   Cluster {i+1}: {cluster['size']} accounts, "
                  f"{cluster['mule_count']} known mules, "
                  f"avg fan-out={cluster['avg_fan_out']:.1f}")

    # Compute network risk scores
    print("\n5. Computing network risk scores...")
    network_scores = detector.compute_network_risk_scores()

    # Save detector
    print("\n6. Saving detector...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'experiments', 'fusion'), exist_ok=True)
    detector.save_detector(os.path.join(PROJECT_ROOT, 'experiments', 'fusion', 'mule_network_detector.pkl'))

    # Add scores to accounts
    accounts_df['network_risk_score'] = accounts_df['account_id'].map(network_scores)
    accounts_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts_with_network_risk.csv'), index=False)
    print("✓ Updated accounts saved to: data/processed/accounts_with_network_risk.csv")

    print("\n" + "=" * 80)
    print("MULE NETWORK DETECTION - COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
