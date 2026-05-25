from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans

from backend.app.database import execute, fetch_all, fetch_one
from backend.app.services.gallery_service import GalleryService
from backend.app.utils.image_utils import thumbnail_to_data_url


CLUSTER_PREVIEW_LIMIT = 12


class ClusterService:
    def __init__(self):
        self.gallery_service = GalleryService()

    def run(self, cluster_count: int, user_id: int | None) -> dict:
        items = self.gallery_service.load_feature_rows(include_thumbnails=False)
        if len(items) < cluster_count:
            raise ValueError("图片数量不足")

        vectors = np.stack([item["feature"] for item in items], axis=0)
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        groups = []
        for cluster_id in range(cluster_count):
            all_cluster_items = [
                {
                    "id": item["id"],
                    "name": item["originalName"],
                    "labelName": item["labelName"],
                    "thumbnailPath": item["thumbnailPath"],
                }
                for index, item in enumerate(items)
                if int(labels[index]) == cluster_id
            ]
            label_distribution = Counter(item.get("labelName") or "未设置" for item in all_cluster_items)
            preview_items = [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "thumbnail": thumbnail_to_data_url(item["thumbnailPath"]),
                    "labelName": item["labelName"],
                }
                for item in all_cluster_items[:CLUSTER_PREVIEW_LIMIT]
            ]
            groups.append(
                {
                    "clusterId": cluster_id + 1,
                    "count": len(all_cluster_items),
                    "items": preview_items,
                    "labelDistribution": [
                        {"label": label, "count": count}
                        for label, count in label_distribution.most_common()
                    ],
                }
            )

        payload = {
            "clusterCount": cluster_count,
            "inertia": round(float(kmeans.inertia_), 4),
            "totalImages": len(items),
            "groups": groups,
        }
        run_id = execute(
            """
            INSERT INTO cluster_runs (cluster_count, inertia_value, total_images, payload_json, created_by)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (cluster_count, float(kmeans.inertia_), len(items), json.dumps(payload), user_id),
        )
        return self.get_run(int(run_id))

    def latest(self) -> dict | None:
        row = fetch_one(
            """
            SELECT cr.*, u.display_name
            FROM cluster_runs cr
            LEFT JOIN users u ON u.id = cr.created_by
            ORDER BY cr.id DESC
            LIMIT 1
            """
        )
        if not row:
            return None
        return self._payload_from_row(row)

    def list_runs(self) -> list[dict]:
        rows = fetch_all(
            """
            SELECT cr.id, cr.cluster_count, cr.inertia_value, cr.total_images, cr.created_at, u.display_name
            FROM cluster_runs cr
            LEFT JOIN users u ON u.id = cr.created_by
            ORDER BY cr.id DESC
            """
        )
        history = []
        for row in rows:
            total_images = int(row.get("total_images") or 0)
            cluster_count = int(row.get("cluster_count") or 0)
            inertia = round(float(row.get("inertia_value") or 0), 4)
            history.append(
                {
                    "runId": int(row["id"]),
                    "runCode": self._run_code(str(row.get("created_at")), int(row["id"])),
                    "clusterCount": cluster_count,
                    "totalImages": total_images,
                    "inertia": inertia,
                    "inertiaPerImage": round(inertia / max(total_images, 1), 4),
                    "averageGroupSize": round(total_images / max(cluster_count, 1), 2),
                    "largestGroup": "--",
                    "smallestGroup": "--",
                    "createdBy": row.get("display_name") or "--",
                    "createdAt": str(row.get("created_at")),
                    "assessment": "已生成",
                }
            )
        return history

    def get_run(self, run_id: int) -> dict:
        row = fetch_one(
            """
            SELECT cr.*, u.display_name
            FROM cluster_runs cr
            LEFT JOIN users u ON u.id = cr.created_by
            WHERE cr.id = %s
            """,
            (run_id,),
        )
        if not row:
            raise ValueError("分析记录不存在")
        return self._payload_from_row(row)

    def _payload_from_row(self, row: dict) -> dict:
        payload = json.loads(row["payload_json"])
        return self._enrich_payload(
            payload,
            run_id=int(row["id"]),
            created_at=str(row["created_at"]),
            created_by=row.get("display_name") or "--",
            total_images=int(row.get("total_images") or 0),
            inertia=float(row.get("inertia_value") or 0),
            cluster_count=int(row.get("cluster_count") or 0),
        )

    def _enrich_payload(
        self,
        payload: dict,
        run_id: int | None = None,
        created_at: str | None = None,
        created_by: str | None = None,
        total_images: int | None = None,
        inertia: float | None = None,
        cluster_count: int | None = None,
    ) -> dict:
        groups = payload.get("groups") or []
        resolved_total = int(total_images or payload.get("totalImages") or sum(int(group.get("count") or 0) for group in groups))
        resolved_inertia = round(float(inertia if inertia is not None else payload.get("inertia") or 0), 4)
        resolved_cluster_count = int(cluster_count or payload.get("clusterCount") or len(groups))
        enriched = {
            **payload,
            "runId": run_id if run_id is not None else payload.get("runId"),
            "createdAt": created_at or payload.get("createdAt"),
            "createdBy": created_by or payload.get("createdBy"),
            "totalImages": resolved_total,
            "clusterCount": resolved_cluster_count,
            "inertia": resolved_inertia,
        }
        enriched["runCode"] = payload.get("runCode") or self._run_code(enriched.get("createdAt"), enriched.get("runId"))
        enriched["report"] = self._build_report(groups, resolved_cluster_count, resolved_total, resolved_inertia)
        return enriched

    def _run_code(self, created_at: str | None, run_id: int | None) -> str:
        if created_at:
            for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return f"JL{datetime.strptime(str(created_at).split('+')[0], pattern).strftime('%Y%m%d%H%M%S')}"
                except ValueError:
                    continue
        return f"JL{int(run_id or 0):014d}"

    def _build_report(self, groups: list[dict], cluster_count: int, total_images: int, inertia: float) -> dict:
        label_counter: Counter[str] = Counter()
        group_counts = []
        group_summaries = []
        for group in groups:
            items = group.get("items") or []
            if group.get("labelDistribution"):
                distribution = Counter(
                    {
                        item.get("label") or "未设置": int(item.get("count") or 0)
                        for item in group.get("labelDistribution") or []
                    }
                )
            else:
                labels = [item.get("labelName") or "未设置" for item in items]
                distribution = Counter(labels)
            label_counter.update(distribution)
            count = int(group.get("count") or len(items))
            group_counts.append(count)
            if distribution:
                dominant_label, dominant_count = distribution.most_common(1)[0]
            else:
                dominant_label, dominant_count = "未设置", 0
            group_summaries.append(
                {
                    "clusterId": group.get("clusterId"),
                    "count": count,
                    "dominantLabel": dominant_label,
                    "dominantCount": dominant_count,
                    "dominantRatio": round(dominant_count / max(count, 1), 4),
                    "labelDistribution": [
                        {
                            "label": label,
                            "count": label_count,
                            "ratio": round(label_count / max(count, 1), 4),
                        }
                        for label, label_count in distribution.most_common(6)
                    ],
                }
            )

        largest_group = max(group_counts, default=0)
        smallest_group = min(group_counts, default=0)
        balance_ratio = round(smallest_group / max(largest_group, 1), 4)
        if total_images == 0:
            assessment = "暂无可分析图片"
        elif balance_ratio >= 0.65:
            assessment = "分组规模较均衡，适合观察图库中的主要视觉类型"
        elif balance_ratio >= 0.35:
            assessment = "分组规模存在差异，可结合分组图片继续判断图库分布"
        else:
            assessment = "分组规模差异较大，说明图库中存在明显主类或少量离散图片"

        return {
            "totalImages": total_images,
            "averageGroupSize": round(total_images / max(cluster_count, 1), 2),
            "largestGroup": largest_group,
            "smallestGroup": smallest_group,
            "balanceRatio": balance_ratio,
            "inertiaPerImage": round(inertia / max(total_images, 1), 4),
            "labelCount": len(label_counter),
            "topLabels": [
                {"label": label, "count": count, "ratio": round(count / max(total_images, 1), 4)}
                for label, count in label_counter.most_common(8)
            ],
            "groupSummaries": group_summaries,
            "assessment": assessment,
        }
