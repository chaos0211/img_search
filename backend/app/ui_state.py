from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from backend.app.config import settings
from backend.app.database import fetch_one
from backend.app.services.auth_service import AuthService
from backend.app.services.cluster_service import ClusterService
from backend.app.services.duplicate_service import DuplicateService
from backend.app.services.gallery_service import GalleryService
from backend.app.services.metric_service import MetricService
from backend.app.services.offline_pipeline_service import OfflinePipelineService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.upload_service import UploadService
from backend.app.utils.image_utils import thumbnail_to_data_url
from backend.app.utils.pagination import paginate

auth_service = AuthService()
gallery_service = GalleryService()
upload_service = UploadService()
retrieval_service = RetrievalService()
duplicate_service = DuplicateService()
cluster_service = ClusterService()
metric_service = MetricService()
offline_pipeline_service = OfflinePipelineService()


def initialize_session(session_state: Any) -> None:
    defaults = {
        "auth_user": None,
        "active_module": "gallery",
        "auth_mode": "login",
        "gallery_tab": "batch",
        "gallery_batch_page": 1,
        "offline_tab": "dataset",
        "upload_tab": "single",
        "retrieval_tab": "image",
        "duplicate_tab": "scan",
        "gallery_page": 1,
        "gallery_filter_batch": "",
        "gallery_filter_label": "",
        "offline_dataset_page": 1,
        "offline_training_page": 1,
        "offline_evaluation_page": 1,
        "offline_model_page": 1,
        "offline_evaluation_model": "current",
        "offline_feature_page": 1,
        "offline_experiment_page": 1,
        "upload_page": 1,
        "retrieval_gallery_page": 1,
        "retrieval_gallery_filter_batch": "",
        "retrieval_gallery_filter_label": "",
        "image_retrieval_result_page": 1,
        "attribute_retrieval_result_page": 1,
        "duplicate_page": 1,
        "cluster_page": 1,
        "cluster_history_page": 1,
        "cluster_detail_page": 1,
        "user_page": 1,
        "top_k": settings.top_k_default,
        "duplicate_threshold": 0.98,
        "duplicate_threshold_eval": None,
        "cluster_count": 5,
        "retrieval_method": "faiss",
        "image_retrieval_result": None,
        "attribute_retrieval_result": None,
        "attribute_recognition_result": None,
        "duplicate_results": [],
        "cluster_result": None,
        "cluster_detail_result": None,
        "notice": None,
        "notice_type": "success",
        "last_event_id": None,
    }
    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def build_state(session_state: Any) -> dict[str, Any]:
    user = session_state["auth_user"]
    notice = session_state.get("notice")
    state = {
        "appTitle": "相似图像检索系统",
        "authenticated": user is not None,
        "refreshIntervalMs": settings.refresh_interval_ms,
        "notice": {"message": notice, "type": session_state.get("notice_type", "success")} if notice else None,
        "authMode": session_state["auth_mode"],
    }
    session_state["notice"] = None

    if not user:
        return state

    gallery_service.ensure_test_batches()
    gallery_service.ensure_label_categories()
    metrics = metric_service.summary()
    retrieval_gallery = gallery_service.list_images_page(
        session_state["retrieval_gallery_page"],
        settings.default_page_size,
        batch_key=session_state.get("retrieval_gallery_filter_batch", ""),
        label_name=session_state.get("retrieval_gallery_filter_label", ""),
    )
    gallery_batches = gallery_service.list_batches()
    offline_dataset = offline_pipeline_service.dataset_status()
    offline_training = offline_pipeline_service.training_status()
    offline_evaluation = offline_pipeline_service.evaluation_status(session_state.get("offline_evaluation_model"))
    session_state["offline_evaluation_model"] = offline_evaluation.get("selectedModel") or "current"
    offline_experiments = offline_pipeline_service.experiment_status()
    active_module = _sanitize_module(session_state["active_module"])
    cluster_payload = {
        "clusterCount": session_state["cluster_count"],
        "result": None,
        "history": paginate([], session_state["cluster_history_page"], settings.user_page_size),
        "detail": None,
    }
    if active_module == "cluster":
        cluster_result = session_state["cluster_result"] or cluster_service.latest()
        session_state["cluster_result"] = cluster_result
        cluster_payload = {
            "clusterCount": session_state["cluster_count"],
            "result": _paginate_cluster(cluster_result, session_state["cluster_page"]),
            "history": paginate(cluster_service.list_runs(), session_state["cluster_history_page"], settings.user_page_size),
            "detail": _paginate_cluster(session_state["cluster_detail_result"], session_state["cluster_detail_page"]),
        }

    state.update(
        {
            "user": user,
            "menu": _menu_items(),
            "activeModule": active_module,
            "gallery": {
                "tab": session_state["gallery_tab"],
                "overview": gallery_service.overview(),
                "filters": {
                    "batchKey": session_state.get("gallery_filter_batch", ""),
                    "labelName": session_state.get("gallery_filter_label", ""),
                },
                "images": gallery_service.list_images_page(
                    session_state["gallery_page"],
                    settings.default_page_size,
                    batch_key=session_state.get("gallery_filter_batch", ""),
                    label_name=session_state.get("gallery_filter_label", ""),
                ),
                "batches": paginate(gallery_batches, session_state["gallery_batch_page"], settings.user_page_size),
                "batchOptions": gallery_batches,
                "testGroups": gallery_service.list_test_group_options(),
                "labelOptions": gallery_service.list_label_options(),
                "labelCategories": gallery_service.list_label_categories(),
            },
            "offline": {
                "tab": session_state["offline_tab"],
                "dataset": {
                    "selector": offline_dataset["selector"],
                    "summary": offline_dataset["summary"],
                    "basicInfo": offline_dataset["basicInfo"],
                    "preprocessScheme": offline_dataset["preprocessScheme"],
                    "outputInfo": offline_dataset["outputInfo"],
                    "classDistribution": offline_dataset["classes"],
                    "classes": paginate(offline_dataset["classes"], session_state["offline_dataset_page"], settings.user_page_size),
                    "manifestOptions": offline_dataset["manifestOptions"],
                },
                "training": {
                    "summary": offline_training["summary"],
                    "modelInfo": offline_training["modelInfo"],
                    "modelArchitecture": offline_training["modelArchitecture"],
                    "trainingScheme": offline_training["trainingScheme"],
                    "runInfo": offline_training["runInfo"],
                    "latest": offline_training["latest"],
                    "history": paginate(offline_training["history"], session_state["offline_training_page"], settings.user_page_size),
                    "checkpointOptions": offline_training["checkpointOptions"],
                    "deviceOptions": offline_training["deviceOptions"],
                    "optimizerOptions": offline_training["optimizerOptions"],
                    "manifestOptions": offline_training["manifestOptions"],
                    "validationManifestOptions": offline_training["validationManifestOptions"],
                    "runtime": offline_training["runtime"],
                },
                "evaluation": {
                    "selectedModel": offline_evaluation["selectedModel"],
                    "modelOptions": offline_evaluation["modelOptions"],
                    "modelList": paginate(offline_evaluation["modelList"], session_state["offline_model_page"], settings.user_page_size),
                    "summary": offline_evaluation["summary"],
                    "history": offline_evaluation["history"],
                    "classNames": offline_evaluation["classNames"],
                    "confusionMatrix": offline_evaluation["confusionMatrix"],
                    "perClassMetrics": paginate(offline_evaluation["perClassMetrics"], session_state["offline_evaluation_page"], settings.user_page_size),
                    "runInfo": offline_evaluation["runInfo"],
                },
                "experiments": {
                    "allRecords": offline_experiments["records"],
                    "records": paginate(offline_experiments["records"], session_state["offline_experiment_page"], settings.user_page_size),
                    "featureManifestOptions": offline_experiments["featureManifestOptions"],
                    "runtime": offline_pipeline_service.experiment_runtime_status(),
                },
            },
            "retrieval": {
                "tab": session_state["retrieval_tab"],
                "method": session_state["retrieval_method"],
                "topK": session_state["top_k"],
                "methods": [
                    {"label": "FlatIP", "value": "faiss"},
                    {"label": "暴力索引", "value": "brute"},
                    {"label": "HNSW", "value": "hnsw"},
                    {"label": "PQ", "value": "pq"},
                    {"label": "KD-Tree", "value": "kd_tree"},
                ],
                "index": retrieval_service.index_status(),
                "gallery": retrieval_gallery,
                "galleryFilters": {
                    "batchKey": session_state.get("retrieval_gallery_filter_batch", ""),
                    "labelName": session_state.get("retrieval_gallery_filter_label", ""),
                },
                "attributeOptions": retrieval_service.attribute_options(),
                "imageResult": _paginate_result(session_state["image_retrieval_result"], session_state["image_retrieval_result_page"]),
                "attributeResult": _paginate_result(session_state["attribute_retrieval_result"], session_state["attribute_retrieval_result_page"]),
            },
            "recognition": {
                "result": session_state["attribute_recognition_result"],
            },
            "duplicate": {
                "tab": session_state["duplicate_tab"],
                "threshold": session_state["duplicate_threshold"],
                "pairs": _paginate_duplicate(session_state["duplicate_results"], session_state["duplicate_page"]),
                "thresholdEval": session_state["duplicate_threshold_eval"],
            },
            "cluster": {
                **cluster_payload,
            },
            "metrics": metrics,
        }
    )

    state["profile"] = {
        "displayName": user["displayName"],
        "username": user["username"],
        "phone": user.get("phone") or "",
        "email": user.get("email") or "",
        "organization": user.get("organization") or "",
    }

    return state


def handle_action(session_state: Any, action: dict[str, Any]) -> None:
    action_type = action.get("type")
    payload = action.get("payload", {})

    if action_type == "setAuthMode":
        session_state["auth_mode"] = payload.get("mode", "login")
        return

    if action_type == "login":
        user = auth_service.login(payload["username"], payload["password"])
        _validate_client_role(payload, user["role"])
        session_state["auth_user"] = user
        session_state["active_module"] = "gallery"
        _set_notice(session_state, "success", "登录成功")
        return

    if action_type == "register":
        _validate_required_fields(payload, ("username", "displayName", "phone", "email", "organization", "password", "confirmPassword"))
        if payload["password"] != payload["confirmPassword"]:
            raise ValueError("两次密码不一致")
        user = auth_service.register(
            payload["username"],
            payload["displayName"],
            payload["phone"],
            payload["email"],
            payload["organization"],
            payload["password"],
        )
        _validate_client_role(payload, user["role"])
        session_state["auth_user"] = user
        session_state["active_module"] = "gallery"
        _set_notice(session_state, "success", "注册成功")
        return

    if action_type == "logout":
        session_state["auth_user"] = None
        session_state["auth_mode"] = "login"
        session_state["image_retrieval_result"] = None
        session_state["attribute_retrieval_result"] = None
        session_state["attribute_recognition_result"] = None
        session_state["duplicate_results"] = []
        session_state["duplicate_threshold_eval"] = None
        session_state["cluster_result"] = None
        session_state["cluster_detail_result"] = None
        _set_notice(session_state, "success", "已退出")
        return

    _require_auth(session_state)

    if action_type == "navigate":
        session_state["active_module"] = _sanitize_module(payload["module"])
        return

    if action_type == "setTab":
        target = payload["target"]
        if target == "offline":
            value = payload["value"]
            session_state["offline_tab"] = value if value in {"dataset", "training", "evaluation"} else "dataset"
        if target == "gallery":
            session_state["gallery_tab"] = payload["value"]
        if target == "upload":
            session_state["upload_tab"] = payload["value"]
        if target == "retrieval":
            session_state["retrieval_tab"] = payload["value"]
        if target == "duplicate":
            value = payload["value"]
            session_state["duplicate_tab"] = value if value in {"scan", "threshold"} else "scan"
        return

    if action_type == "setPage":
        page_mapping = {
            "offlineDataset": "offline_dataset_page",
            "offlineTraining": "offline_training_page",
            "offlineEvaluation": "offline_evaluation_page",
            "offlineModels": "offline_model_page",
            "offlineExperiments": "offline_experiment_page",
            "gallery": "gallery_page",
            "galleryBatches": "gallery_batch_page",
            "upload": "upload_page",
            "retrievalGallery": "retrieval_gallery_page",
            "imageRetrievalResult": "image_retrieval_result_page",
            "attributeRetrievalResult": "attribute_retrieval_result_page",
            "duplicate": "duplicate_page",
            "cluster": "cluster_page",
            "clusterHistory": "cluster_history_page",
            "clusterDetail": "cluster_detail_page",
        }
        state_key = page_mapping.get(payload["target"])
        if state_key:
            session_state[state_key] = int(payload["page"])
        return

    if action_type == "setEvaluationModel":
        session_state["offline_evaluation_model"] = payload.get("value") or "current"
        session_state["offline_evaluation_page"] = 1
        return

    if action_type == "deleteModelWeight":
        offline_pipeline_service.delete_model_weight(payload.get("value"))
        session_state["offline_model_page"] = 1
        session_state["offline_evaluation_model"] = "current"
        _set_notice(session_state, "success", "模型已删除")
        return

    if action_type == "uploadImage":
        upload_service.upload_image(
            data_url=payload["dataUrl"],
            original_name=payload["originalName"],
            creator_id=session_state["auth_user"]["id"],
            source="upload",
            label_name=payload.get("labelName") or None,
        )
        _set_notice(session_state, "success", "上传成功")
        return

    if action_type == "uploadImages":
        files = list(payload.get("files") or [])
        if not files:
            raise ValueError("请选择图片")
        uploaded = 0
        for item in files:
            upload_service.upload_image(
                data_url=item["dataUrl"],
                original_name=item["originalName"],
                creator_id=session_state["auth_user"]["id"],
                source="upload",
                label_name=item.get("labelName") or payload.get("labelName") or None,
                split_name=item.get("splitName") or payload.get("splitName") or None,
            )
            uploaded += 1
        session_state["gallery_page"] = 1
        _set_notice(session_state, "success", f"上传{uploaded}张")
        return

    if action_type == "importTestGroup":
        result = gallery_service.import_test_group(
            group_name=payload.get("groupName") or "",
            creator_id=session_state["auth_user"]["id"],
            skip_existing=bool(payload.get("skipExisting", True)),
        )
        session_state["gallery_page"] = 1
        _set_notice(session_state, "success", f"{result['groupName']}导入{result['imported']}张，跳过{result['skipped']}张")
        return

    if action_type == "setGalleryFilters":
        session_state["gallery_filter_batch"] = str(payload.get("batchKey") or "")
        session_state["gallery_filter_label"] = str(payload.get("labelName") or "")
        session_state["gallery_page"] = 1
        return

    if action_type == "setRetrievalGalleryFilters":
        session_state["retrieval_gallery_filter_batch"] = str(payload.get("batchKey") or "")
        session_state["retrieval_gallery_filter_label"] = str(payload.get("labelName") or "")
        session_state["retrieval_gallery_page"] = 1
        return

    if action_type == "importCifar":
        result = _import_cifar(
            payload.get("limit", 100),
            payload.get("split", "train"),
            session_state["auth_user"]["id"],
            int(payload.get("startIndex", 0) or 0),
            bool(payload.get("skipExisting", True)),
        )
        _set_notice(session_state, "success", f"导入{result['imported']}张，跳过{result['skipped']}张")
        return

    if action_type == "prepareDataset":
        dataset_name = (payload.get("datasetName") or "cifar10").strip().lower()
        if dataset_name != "cifar10":
            raise ValueError("当前仅支持 CIFAR-10")
        offline_pipeline_service.prepare_dataset(
            train_limit=int(payload["trainLimit"]),
            test_limit=int(payload["testLimit"]),
            gallery_per_class=int(payload["galleryPerClass"]),
            query_per_class=int(payload["queryPerClass"]),
            clean_existing=bool(payload.get("cleanExisting")),
        )
        session_state["offline_dataset_page"] = 1
        _set_notice(session_state, "success", "数据集已更新")
        return

    if action_type == "trainEmbeddingModel":
        offline_pipeline_service.start_training(
            train_manifest_path=payload["trainManifest"],
            validation_manifest_path=payload.get("validationManifest") or None,
            epochs=int(payload["epochs"]),
            early_stop_patience=int(payload.get("earlyStopPatience", 10)),
            batch_size=int(payload["batchSize"]),
            num_workers=int(payload.get("numWorkers", 0)),
            learning_rate=float(payload["learningRate"]),
            optimizer_name=payload.get("optimizerName", "adam"),
            seed=int(payload.get("seed", 42)),
            save_best_only=bool(payload.get("saveBestOnly", True)),
            freeze_backbone=bool(payload.get("freezeBackbone", True)),
            device_name=payload.get("deviceName", "auto"),
        )
        session_state["offline_training_page"] = 1
        session_state["offline_evaluation_page"] = 1
        _set_notice(session_state, "success", "训练已启动")
        return

    if action_type == "stopEmbeddingModel":
        offline_pipeline_service.stop_training()
        _set_notice(session_state, "success", "已发送停止请求")
        return

    if action_type == "runOfflineExperiment":
        offline_pipeline_service.start_experiment(
            experiment_name=payload["experimentName"],
            top_k=int(payload.get("topK") or 10),
            gallery_manifest=payload.get("galleryManifest"),
            query_manifest=payload.get("queryManifest"),
            baseline_gallery_manifest=payload.get("baselineGalleryManifest"),
            baseline_query_manifest=payload.get("baselineQueryManifest"),
            embedding_gallery_manifest=payload.get("embeddingGalleryManifest"),
            embedding_query_manifest=payload.get("embeddingQueryManifest"),
            feature_scheme=payload.get("featureScheme"),
            index_method=payload.get("indexMethod"),
            rerank_enabled=bool(payload.get("rerankEnabled")),
        )
        session_state["offline_experiment_page"] = 1
        _set_notice(session_state, "success", "评估已启动")
        return

    if action_type == "stopOfflineExperiment":
        offline_pipeline_service.stop_experiment()
        _set_notice(session_state, "success", "评估已停止")
        return

    if action_type == "deleteImage":
        gallery_service.delete_image(int(payload["imageId"]))
        _set_notice(session_state, "success", "删除成功")
        return

    if action_type == "deleteGalleryBatch":
        result = gallery_service.delete_batch(
            source=str(payload.get("source") or ""),
            split_name=str(payload.get("splitName") or ""),
        )
        session_state["gallery_page"] = 1
        _set_notice(session_state, "success", f"删除{result['deleted']}张")
        return

    if action_type == "deleteGalleryBatches":
        result = gallery_service.delete_batches(list(payload.get("batches") or []))
        session_state["gallery_page"] = 1
        session_state["gallery_batch_page"] = 1
        _set_notice(session_state, "success", f"删除{result['deleted']}张")
        return

    if action_type == "createLabelCategory":
        gallery_service.create_label_category(payload.get("name") or "")
        _set_notice(session_state, "success", "分类已新增")
        return

    if action_type == "updateLabelCategory":
        gallery_service.update_label_category(int(payload.get("id")), payload.get("name") or "")
        session_state["gallery_page"] = 1
        _set_notice(session_state, "success", "分类已更新")
        return

    if action_type == "deleteLabelCategory":
        gallery_service.delete_label_category(int(payload.get("id")))
        session_state["gallery_page"] = 1
        _set_notice(session_state, "success", "分类已删除")
        return

    if action_type == "searchGallery":
        session_state["retrieval_method"] = payload["method"]
        session_state["top_k"] = int(payload["topK"])
        session_state["image_retrieval_result_page"] = 1
        session_state["image_retrieval_result"] = retrieval_service.search_by_gallery_image(
            image_id=int(payload["imageId"]),
            method=payload["method"],
            top_k=int(payload["topK"]),
            user_id=session_state["auth_user"]["id"],
            feature_type=str(payload.get("featureType") or "none"),
            rerank_enabled=bool(payload.get("rerankEnabled")),
        )
        _set_notice(session_state, "success", "检索完成")
        return

    if action_type == "searchUpload":
        session_state["retrieval_method"] = payload["method"]
        session_state["top_k"] = int(payload["topK"])
        session_state["image_retrieval_result_page"] = 1
        session_state["image_retrieval_result"] = retrieval_service.search_by_upload(
            data_url=payload["dataUrl"],
            original_name=payload["originalName"],
            method=payload["method"],
            top_k=int(payload["topK"]),
            user_id=session_state["auth_user"]["id"],
            feature_type=str(payload.get("featureType") or "none"),
            rerank_enabled=bool(payload.get("rerankEnabled")),
        )
        _set_notice(session_state, "success", "检索完成")
        return

    if action_type == "searchUploadUrl":
        session_state["retrieval_method"] = payload["method"]
        session_state["top_k"] = int(payload["topK"])
        session_state["image_retrieval_result_page"] = 1
        session_state["image_retrieval_result"] = retrieval_service.search_by_url(
            image_url=payload["imageUrl"],
            method=payload["method"],
            top_k=int(payload["topK"]),
            user_id=session_state["auth_user"]["id"],
            feature_type=str(payload.get("featureType") or "none"),
            rerank_enabled=bool(payload.get("rerankEnabled")),
        )
        _set_notice(session_state, "success", "检索完成")
        return

    if action_type == "recognizeAttributes":
        if payload.get("imageUrl"):
            session_state["attribute_recognition_result"] = retrieval_service.recognize_by_url(payload["imageUrl"])
        else:
            session_state["attribute_recognition_result"] = retrieval_service.recognize_by_upload(
                data_url=payload["dataUrl"],
                original_name=payload["originalName"],
            )
        _set_notice(session_state, "success", "识别完成")
        return

    if action_type == "searchAttributes":
        session_state["retrieval_method"] = "faiss"
        session_state["top_k"] = int(payload["topK"])
        session_state["attribute_retrieval_result_page"] = 1
        session_state["attribute_retrieval_result"] = retrieval_service.search_by_attributes(
            attribute_values=list(payload.get("attributes") or []),
            top_k=int(payload["topK"]),
            user_id=session_state["auth_user"]["id"],
            search_mode=str(payload.get("searchMode") or "hybrid"),
        )
        _set_notice(session_state, "success", "检索完成")
        return

    if action_type == "rebuildVectorIndex":
        result = retrieval_service.rebuild_index()
        _set_notice(session_state, "success", f"索引已更新{result.get('vectorCount', 0)}张")
        return

    if action_type == "refreshGalleryFeatures":
        feature_result = gallery_service.rebuild_features()
        index_result = retrieval_service.rebuild_index()
        _set_notice(
            session_state,
            "success",
            f"特征已刷新{feature_result.get('updated', 0)}张，索引{index_result.get('vectorCount', 0)}张",
        )
        return

    if action_type == "runDuplicate":
        session_state["duplicate_threshold"] = float(payload["threshold"])
        session_state["duplicate_page"] = 1
        session_state["duplicate_results"] = duplicate_service.scan(float(payload["threshold"]))
        _set_notice(session_state, "success", "扫描完成")
        return

    if action_type == "runDuplicateThresholdEval":
        session_state["duplicate_threshold_eval"] = duplicate_service.evaluate_thresholds(
            top_k=int(payload.get("topK") or 10),
            sample_size=int(payload.get("sampleSize") or 100),
        )
        _set_notice(session_state, "success", "评估完成")
        return

    if action_type == "deleteDuplicate":
        duplicate_service.delete_duplicate(
            primary_image_id=int(payload["primaryImageId"]),
            duplicate_image_id=int(payload["duplicateImageId"]),
            similarity=float(payload["similarity"]),
            threshold=float(payload["threshold"]),
            user_id=session_state["auth_user"]["id"],
        )
        session_state["duplicate_results"] = duplicate_service.scan(float(payload["threshold"]))
        _set_notice(session_state, "success", "处理完成")
        return

    if action_type == "runCluster":
        session_state["cluster_count"] = int(payload["clusterCount"])
        session_state["cluster_page"] = 1
        session_state["cluster_history_page"] = 1
        session_state["cluster_detail_page"] = 1
        session_state["cluster_detail_result"] = None
        session_state["cluster_result"] = cluster_service.run(int(payload["clusterCount"]), session_state["auth_user"]["id"])
        _set_notice(session_state, "success", "聚类完成")
        return

    if action_type == "openClusterRun":
        session_state["cluster_detail_page"] = 1
        session_state["cluster_detail_result"] = cluster_service.get_run(int(payload["runId"]))
        return

    if action_type == "closeClusterRun":
        session_state["cluster_detail_result"] = None
        session_state["cluster_detail_page"] = 1
        return

    if action_type == "updateProfile":
        _validate_required_fields(payload, ("displayName", "phone", "email", "organization"))
        user = auth_service.update_profile(
            session_state["auth_user"]["id"],
            payload["displayName"],
            payload["phone"],
            payload["email"],
            payload["organization"],
        )
        session_state["auth_user"] = user
        _set_notice(session_state, "success", "保存成功")
        return


def _menu_items() -> list[dict[str, str]]:
    return [
        {"key": "gallery", "label": "图库管理"},
        {"key": "duplicate", "label": "重复度检测"},
        {"key": "retrieval", "label": "检索图片"},
        {"key": "cluster", "label": "聚类管理"},
        {"key": "offline", "label": "模型管理"},
        {"key": "profile", "label": "个人中心"},
    ]


def _sanitize_module(module: str) -> str:
    allowed = {item["key"] for item in _menu_items()}
    return module if module in allowed else "gallery"


def _require_auth(session_state: Any) -> None:
    if not session_state.get("auth_user"):
        raise ValueError("请先登录")


def _validate_client_role(payload: dict[str, Any], role: str) -> None:
    client_role = str(payload.get("__clientRole") or "").strip()
    if client_role and client_role != role:
        raise ValueError("当前入口不允许该角色登录")


def _set_notice(session_state: Any, notice_type: str, message: str) -> None:
    session_state["notice_type"] = notice_type
    session_state["notice"] = message


def _validate_required_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("请完整填写信息")


def _paginate_result(result: dict[str, Any] | None, page: int):
    if not result:
        return None
    paged = paginate(result["results"], page, settings.default_page_size)
    return {
        "query": result["query"],
        "metrics": result["metrics"],
        "recognition": result.get("recognition"),
        "list": paged,
    }


def _paginate_duplicate(results: list[dict[str, Any]], page: int):
    paged = paginate(results, page, settings.duplicate_page_size)
    image_ids = {
        int(image_id)
        for item in results
        for image_id in (item["left"]["id"], item["right"]["id"])
    }
    paged["items"] = [
        {
            **item,
            "left": {
                "id": item["left"]["id"],
                "name": item["left"]["name"],
                "thumbnail": thumbnail_to_data_url(item["left"]["thumbnailPath"]),
            },
            "right": {
                "id": item["right"]["id"],
                "name": item["right"]["name"],
                "thumbnail": thumbnail_to_data_url(item["right"]["thumbnailPath"]),
            },
        }
        for item in paged["items"]
    ]
    paged["pairCount"] = len(results)
    paged["imageCount"] = len(image_ids)
    return paged


def _paginate_cluster(result: dict[str, Any] | None, page: int):
    if not result:
        return None
    preview_groups = []
    for group in result["groups"]:
        preview_groups.append(
            {
                **group,
                "items": list(group.get("items") or [])[:12],
            }
        )
    paged = paginate(preview_groups, page, 4)
    return {
        "runId": result.get("runId"),
        "runCode": result.get("runCode"),
        "clusterCount": result["clusterCount"],
        "totalImages": result.get("totalImages"),
        "inertia": result["inertia"],
        "createdBy": result.get("createdBy"),
        "groups": paged,
        "report": result.get("report"),
        "createdAt": result.get("createdAt"),
    }


def _import_cifar(limit: int, split: str, creator_id: int, start_index: int = 0, skip_existing: bool = True) -> dict[str, int]:
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(settings.project_root / "data" / "raw" / "cifar10"), train=split == "train", download=True)
    limit = max(0, min(int(limit), len(dataset)))
    start_index = max(0, min(int(start_index), len(dataset)))
    imported = 0
    skipped = 0
    for index in range(start_index, len(dataset)):
        if imported >= limit:
            break
        original_name = f"cifar10_{split}_{index}.png"
        if skip_existing and _cifar_exists(original_name, split):
            skipped += 1
            continue
        image, label = dataset[index]
        data_url = _image_to_data_url(image)
        upload_service.upload_image(
            data_url=data_url,
            original_name=original_name,
            creator_id=creator_id,
            source="cifar10",
            label_name=dataset.classes[label],
            split_name=split,
        )
        imported += 1
    return {"imported": imported, "skipped": skipped}


def _cifar_exists(original_name: str, split: str) -> bool:
    row = fetch_one(
        """
        SELECT id FROM images
        WHERE original_name = %s AND source = 'cifar10' AND split_name = %s AND is_deleted = 0
        LIMIT 1
        """,
        (original_name, split),
    )
    return bool(row)


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
