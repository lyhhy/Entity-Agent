"""
Use OfflineHyperEdgeStore to load and inspect 5adf043b hyperedges.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.offline_hyperedge_store import OfflineHyperEdgeStore

store = OfflineHyperEdgeStore.from_parquet(str(ROOT / "eval/offline_hyperedges/hyperedges.parquet"))
print(f"Stats: {store.stats()}")

target_qid = "5adf043b5542993a75d263d4"
edges = store.get_hyperedges_for_question(target_qid)
print(f"\nOffline hyperedges for {target_qid}: {len(edges)}")
for he in edges[:10]:
    print(f"  id={he.get('he_id')} event={he.get('event_name')!r}")
    print(f"    containers: {he.get('container_ids', [])}")
    print(f"    triggers: {he.get('attribute_triggers', {})}")
    print()
