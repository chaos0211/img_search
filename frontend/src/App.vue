<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue"

import bgImage from "./assets/login-bg.svg"
import { apiUrl, backendState, fetchApiJson, sendEvent } from "./streamlit"

type GalleryUploadRow = {
  id: string
  file: File
  name: string
  splitName: string
  labelName: string
}

const state = computed(() => backendState.value)
const authMode = ref<"login" | "register">("login")
const evaluationDetailOpen = ref(false)
const categoryDialogOpen = ref(false)
const selectedRetrievalImageId = ref<number | null>(null)
const selectedRetrievalImageSnapshot = ref<Record<string, any> | null>(null)
const retrievalUploadPreview = ref("")
const recognitionPreview = ref("")
const galleryUploadRows = ref<GalleryUploadRow[]>([])
const selectedGalleryUploadRowIds = ref<string[]>([])
const selectedGalleryBatchKeys = ref<string[]>([])
const testGroupDropdownOpen = ref(false)
const galleryFilterBatchDropdownOpen = ref(false)
const galleryFilterLabelDropdownOpen = ref(false)
const selectedAttributeValues = ref<string[]>([])
const imageQueryMode = ref<"upload" | "gallery">("upload")
const offlineExperimentLocalPage = ref(1)
const offlineExperimentRuntimeOverride = ref<Record<string, any> | null>(null)
const offlineExperimentStarting = ref(false)
const offlineExperimentStartAt = ref(0)
const trainingRuntimeStatus = computed(() => state.value.offline?.training?.runtime?.status ?? "idle")
const trainingIsRunning = computed(() => Boolean(state.value.offline?.training?.runtime?.isRunning))
const chartFrame = {
  width: 540,
  height: 220,
  left: 48,
  right: 18,
  top: 24,
  bottom: 36,
}

const loginForm = reactive({
  username: "user",
  password: "123456",
})

const registerForm = reactive({
  username: "",
  displayName: "",
  phone: "",
  email: "",
  organization: "",
  password: "",
  confirmPassword: "",
})

const retrievalGalleryForm = reactive({
  method: "faiss",
  featureType: "none",
  rerankEnabled: false,
  topK: 10,
})

const retrievalGalleryFilterForm = reactive({
  batchKey: "",
  labelName: "",
})

const retrievalUploadForm = reactive({
  file: null as File | null,
  imageUrl: "",
  method: "faiss",
  featureType: "none",
  rerankEnabled: false,
  topK: 10,
})

const galleryImportForm = reactive({
  groupName: "test1",
  skipExisting: true,
})

const galleryFilterForm = reactive({
  batchKey: "",
  labelName: "",
})
const categoryForm = reactive({
  id: null as number | null,
  name: "",
})
const pageJumpInputs = reactive<Record<string, string>>({})

const recognitionForm = reactive({
  file: null as File | null,
  imageUrl: "",
})

const attributeSearchForm = reactive({
  searchMode: "hybrid",
  topK: 20,
})

const duplicateThreshold = ref(0.98)
const duplicatePresetValue = ref(0.98)
const duplicateEvalTopK = ref(10)
const duplicateEvalSampleSize = ref(100)
const clusterCount = ref(5)
const duplicateThresholdPresets = [
  { label: "严格", value: 0.98 },
  { label: "推荐", value: 0.95 },
  { label: "宽松", value: 0.9 },
]
const datasetSeries = [
  { key: "trainCount", label: "训练集", className: "bar-train" },
  { key: "testCount", label: "测试集", className: "bar-test" },
  { key: "galleryCount", label: "实验库", className: "bar-gallery" },
  { key: "queryCount", label: "查询集", className: "bar-query" },
]
const offlineTrainingForm = reactive({
  trainManifest: "data/manifests/cifar10_train.csv",
  validationManifest: "data/manifests/cifar10_test.csv",
  epochs: 10,
  earlyStopPatience: 3,
  batchSize: 16,
  numWorkers: 2,
  learningRate: 0.001,
  optimizerName: "adam",
  seed: 42,
  saveBestOnly: true,
  freezeBackbone: true,
  deviceName: "",
})
const offlineExperimentForm = reactive({
  evaluationGoal: "matrix",
  experimentName: "matrix",
  featureScheme: "baseline",
  indexMethod: "all",
  rerankEnabled: false,
  topK: 10,
  historyKey: "",
  galleryManifest: "features/cifar10_gallery_embedding.csv",
  queryManifest: "features/cifar10_query_embedding.csv",
  baselineGalleryManifest: "features/cifar10_gallery_baseline.csv",
  baselineQueryManifest: "features/cifar10_query_baseline.csv",
  embeddingGalleryManifest: "features/cifar10_gallery_embedding.csv",
  embeddingQueryManifest: "features/cifar10_query_embedding.csv",
})
const profileForm = reactive({
  displayName: "",
  phone: "",
  email: "",
  organization: "",
})
const uiState = reactive({
  initialized: false,
  username: "",
  activeModule: "gallery",
  galleryTab: "batch",
  offlineTab: "dataset",
  retrievalTab: "image",
  duplicateTab: "scan",
  clusterTab: "manage",
})
const toast = ref<{ message: string; type: string } | null>(null)
const duplicateRefreshing = ref(false)
const hiddenDuplicatePairKeys = ref<string[]>([])
let toastTimer: number | undefined
let trainingRefreshTimer: number | undefined
let experimentRefreshTimer: number | undefined

watch(
  () => state.value.user?.username ?? "",
  (username) => {
    if (!username) {
      uiState.initialized = false
      uiState.username = ""
      uiState.activeModule = "gallery"
      uiState.galleryTab = "batch"
      uiState.offlineTab = "dataset"
      uiState.retrievalTab = "image"
      uiState.duplicateTab = "scan"
      uiState.clusterTab = "manage"
      return
    }

    if (!uiState.initialized || uiState.username !== username) {
      uiState.initialized = true
      uiState.username = username
      uiState.activeModule = state.value.activeModule ?? "gallery"
      uiState.galleryTab = state.value.gallery?.tab ?? "batch"
      uiState.offlineTab = ["dataset", "training", "evaluation"].includes(state.value.offline?.tab ?? "") ? String(state.value.offline?.tab) : "dataset"
      uiState.retrievalTab = state.value.retrieval?.tab ?? "image"
      uiState.duplicateTab = state.value.duplicate?.tab ?? "scan"
      uiState.clusterTab = "manage"
    }
  },
  { immediate: true },
)

watch(
  () => state.value.authMode,
  (mode) => {
    authMode.value = mode === "register" ? "register" : "login"
  },
  { immediate: true },
)

watch(
  () => state.value.notice,
  (notice) => {
    if (!notice?.message) return
    toast.value = notice
    window.clearTimeout(toastTimer)
    toastTimer = window.setTimeout(() => {
      toast.value = null
    }, 2200)
  },
  { deep: true, immediate: true },
)

watch(
  () => state.value.retrieval,
  (retrieval) => {
    if (!retrieval) return
    retrievalGalleryForm.method = retrieval.method ?? "faiss"
    retrievalGalleryForm.topK = retrieval.topK ?? 10
    retrievalGalleryFilterForm.batchKey = retrieval.galleryFilters?.batchKey ?? ""
    retrievalGalleryFilterForm.labelName = retrieval.galleryFilters?.labelName ?? ""
    retrievalUploadForm.method = retrieval.method ?? "faiss"
    retrievalUploadForm.topK = retrieval.topK ?? 10
  },
  { deep: true, immediate: true },
)

watch(
  () => state.value.duplicate,
  (duplicate) => {
    if (duplicate?.threshold) {
      duplicateThreshold.value = duplicate.threshold
      const matchedPreset = duplicateThresholdPresets.find((item) => Number(item.value) === Number(duplicate.threshold))
      if (matchedPreset) {
        duplicatePresetValue.value = matchedPreset.value
      }
    }
  },
  { deep: true, immediate: true },
)

watch(
  () => state.value.cluster,
  (cluster) => {
    if (cluster?.clusterCount) {
      clusterCount.value = cluster.clusterCount
    }
  },
  { deep: true, immediate: true },
)

watch(
  () => state.value.gallery?.filters,
  (filters) => {
    galleryFilterForm.batchKey = filters?.batchKey ?? ""
    galleryFilterForm.labelName = filters?.labelName ?? ""
  },
  { deep: true, immediate: true },
)

watch(
  () => state.value.offline,
  (offline) => {
    const manifestOptions = offline?.training?.manifestOptions ?? []
    const validationManifestOptions = offline?.training?.validationManifestOptions ?? []
    const deviceOptions = offline?.training?.deviceOptions ?? []
    const optimizerOptions = offline?.training?.optimizerOptions ?? []
    if (manifestOptions.length && !manifestOptions.includes(offlineTrainingForm.trainManifest)) {
      offlineTrainingForm.trainManifest = manifestOptions[0]
    }
    if (validationManifestOptions.length && !validationManifestOptions.includes(offlineTrainingForm.validationManifest)) {
      offlineTrainingForm.validationManifest = validationManifestOptions[0]
    }
    const availableDeviceOptions = deviceOptions.filter((item: { available?: boolean }) => item.available !== false)
    if (
      availableDeviceOptions.length
      && !availableDeviceOptions.some((item: { value: string }) => item.value === offlineTrainingForm.deviceName)
    ) {
      offlineTrainingForm.deviceName = availableDeviceOptions[0].value
    }
    if (optimizerOptions.length && !optimizerOptions.some((item: { value: string }) => item.value === offlineTrainingForm.optimizerName)) {
      offlineTrainingForm.optimizerName = optimizerOptions[0].value
    }

    const featureManifestOptions = offline?.experiments?.featureManifestOptions ?? []
    syncOfflineExperimentManifests(featureManifestOptions)
  },
  { deep: true, immediate: true },
)

watch(
  () => [offlineExperimentForm.featureScheme, offlineExperimentForm.rerankEnabled],
  () => {
    const options = state.value.offline?.experiments?.featureManifestOptions ?? []
    syncOfflineExperimentManifests(options, true)
    offlineExperimentForm.historyKey = ""
    offlineExperimentLocalPage.value = 1
  },
)

watch(
  () => state.value.offline?.training?.runtime?.isRunning,
  (isRunning) => {
    window.clearInterval(trainingRefreshTimer)
    if (isRunning) {
      trainingRefreshTimer = window.setInterval(() => {
        sendEvent("refresh")
      }, 3000)
    }
  },
  { immediate: true },
)

watch(
  () => state.value.profile,
  (profile) => {
    profileForm.displayName = profile?.displayName ?? state.value.user?.displayName ?? ""
    profileForm.phone = profile?.phone ?? state.value.user?.phone ?? ""
    profileForm.email = profile?.email ?? state.value.user?.email ?? ""
    profileForm.organization = profile?.organization ?? state.value.user?.organization ?? ""
  },
  { deep: true, immediate: true },
)

const activeModule = computed(() => uiState.activeModule)
const activeGalleryTab = computed(() => uiState.galleryTab)
const activeOfflineTab = computed(() => (["dataset", "training", "evaluation"].includes(uiState.offlineTab) ? uiState.offlineTab : "dataset"))
const activeRetrievalTab = computed(() => (["image", "evaluation"].includes(uiState.retrievalTab) ? uiState.retrievalTab : "image"))
const activeDuplicateTab = computed(() => (["scan", "threshold"].includes(uiState.duplicateTab) ? uiState.duplicateTab : "scan"))
const activeClusterTab = computed(() => uiState.clusterTab)
const featureManifestOptions = computed(() => state.value.offline?.experiments?.featureManifestOptions ?? [])
const isFeatureComparisonExperiment = computed(() => offlineExperimentForm.evaluationGoal === "feature_compare")
const offlineExperimentRuntime = computed(() => offlineExperimentRuntimeOverride.value ?? state.value.offline?.experiments?.runtime ?? {})
const offlineExperimentRunning = computed(() => Boolean(offlineExperimentRuntime.value?.isRunning))

watch(
  () => [state.value.authenticated, uiState.activeModule],
  ([authenticated, module]) => {
    if (authenticated && module && module !== state.value.activeModule) {
      void sendEvent("navigate", { module: String(module) })
    }
  },
  { immediate: true },
)
const offlineExperimentBusy = computed(() => offlineExperimentStarting.value || offlineExperimentRunning.value)
const offlineExperimentProgress = computed(() => {
  const runtime = offlineExperimentRuntime.value ?? {}
  if (runtime.status === "completed") return 100
  if (["failed", "cancelled", "idle"].includes(String(runtime.status ?? ""))) return 0
  return Math.max(0, Math.min(99, Number(runtime.progress ?? 0)))
})
const offlineExperimentStatusText = computed(() => {
  const runtime = offlineExperimentRuntime.value ?? {}
  const status = String(runtime.status ?? "")
  if (status === "failed") return String(runtime.message || "评估失败")
  if (status === "cancelled") return "已停止"
  if (status === "running" && runtime.message) return String(runtime.message)
  return ""
})
const offlineEvaluationGalleryCount = computed(() => Number(state.value.offline?.dataset?.summary?.preparedTrainCount ?? 0))
async function refreshOfflineExperimentRuntime() {
  try {
    const runtime = await fetchApiJson("/api/offline-experiment-runtime")
    if (!runtime?.isRunning && offlineExperimentStarting.value && Date.now() - offlineExperimentStartAt.value < 8000) {
      return
    }
    offlineExperimentStarting.value = false
    offlineExperimentRuntimeOverride.value = runtime
    if (!runtime?.isRunning) {
      window.clearInterval(experimentRefreshTimer)
      experimentRefreshTimer = undefined
      sendEvent("refresh")
    }
  } catch {
    offlineExperimentStarting.value = false
    window.clearInterval(experimentRefreshTimer)
    experimentRefreshTimer = undefined
  }
}
watch(
  () => offlineExperimentBusy.value,
  (isRunning) => {
    window.clearInterval(experimentRefreshTimer)
    if (isRunning) {
      void refreshOfflineExperimentRuntime()
      experimentRefreshTimer = window.setInterval(() => {
        void refreshOfflineExperimentRuntime()
      }, 1200)
    }
  },
  { immediate: true },
)
const offlineExperimentRecordPrefixes = computed(() => {
  return ["matrix_"]
})
const filteredOfflineExperimentRecords = computed(() => {
  const records = state.value.offline?.experiments?.allRecords ?? state.value.offline?.experiments?.records?.items ?? []
  const prefixes = offlineExperimentRecordPrefixes.value
  const matrixRecords = records.filter((item: Record<string, any>) =>
    prefixes.some((prefix) => String(item.name ?? "").startsWith(prefix)) &&
    String(item.featureScheme ?? "") === offlineExperimentForm.featureScheme,
  ).filter((item: Record<string, any>) =>
    String(item.indexType ?? "") !== "faiss",
  )
  const selectedRunId = offlineExperimentForm.historyKey || offlineExperimentHistoryOptions.value[0]?.value || ""
  if (!selectedRunId) return matrixRecords
  return matrixRecords.filter((item: Record<string, any>) => String(item.runId ?? "") === selectedRunId)
})
const offlineExperimentHistoryOptions = computed(() => {
  const records = state.value.offline?.experiments?.allRecords ?? state.value.offline?.experiments?.records?.items ?? []
  const prefixes = offlineExperimentRecordPrefixes.value
  const seen = new Set<string>()
  return records
    .filter((item: Record<string, any>) =>
      prefixes.some((prefix) => String(item.name ?? "").startsWith(prefix)) &&
      String(item.featureScheme ?? "") === offlineExperimentForm.featureScheme,
    )
    .filter((item: Record<string, any>) =>
      String(item.indexType ?? "") !== "faiss",
    )
    .map((item: Record<string, any>) => ({
      value: String(item.runId ?? ""),
      label: String(item.runLabel ?? item.name ?? ""),
      createdAt: String(item.createdAt ?? item.updatedAt ?? ""),
    }))
    .filter((item) => {
      if (!item.value || seen.has(item.value)) return false
      seen.add(item.value)
      return true
    })
})
watch(offlineExperimentHistoryOptions, (options) => {
  if (!options.length) {
    offlineExperimentForm.historyKey = ""
    return
  }
  if (!options.some((item) => item.value === offlineExperimentForm.historyKey)) {
    offlineExperimentForm.historyKey = options[0].value
  }
}, { immediate: true })
const offlineExperimentPageSize = 10
const offlineExperimentTotalPages = computed(() => Math.max(1, Math.ceil(filteredOfflineExperimentRecords.value.length / offlineExperimentPageSize)))
const visibleOfflineExperimentRecords = computed(() => {
  const page = Math.min(Math.max(1, offlineExperimentLocalPage.value), offlineExperimentTotalPages.value)
  const start = (page - 1) * offlineExperimentPageSize
  return filteredOfflineExperimentRecords.value.slice(start, start + offlineExperimentPageSize)
})
const offlineEvaluationCharts = [
  { key: "mapAtK", title: "mAP@K", format: "percent", color: "#0d4b5b" },
  { key: "recallAtK", title: "Recall@K", format: "percent", color: "#36a3a3" },
  { key: "precisionAtK", title: "Precision@K", format: "percent", color: "#d98c2b" },
  { key: "elapsedMs", title: "检索延时", format: "ms", color: "#e59a34" },
  { key: "indexSizeBytes", title: "索引大小", format: "bytes", color: "#6d58d8" },
] as const
watch(offlineExperimentTotalPages, (totalPages) => {
  if (offlineExperimentLocalPage.value > totalPages) {
    offlineExperimentLocalPage.value = totalPages
  }
})
const menuItems = computed(() => state.value.menu ?? [])
const galleryImages = computed(() => state.value.gallery?.images?.items ?? [])
const galleryBatches = computed(() => state.value.gallery?.batches?.items ?? [])
const galleryLabelOptions = computed(() => state.value.gallery?.labelOptions ?? [])
const galleryLabelCategories = computed(() => state.value.gallery?.labelCategories ?? [])
const galleryBatchOptions = computed(() =>
  (state.value.gallery?.batchOptions ?? []).map((item: Record<string, any>) => ({
    label: `${item.label}（${sourceText(item.source)}，${item.imageCount}张）`,
    value: `${item.source}||${item.splitName ?? ""}`,
    source: item.source,
    splitName: item.splitName ?? "",
  })),
)
const galleryUploadBatchOptions = computed(() => [
  { label: "本地上传", value: "" },
  ...(state.value.gallery?.testGroups ?? []).map((item: Record<string, any>) => ({
    label: item.label,
    value: item.value,
  })),
])
const selectedTestGroup = computed(() =>
  (state.value.gallery?.testGroups ?? []).find((item: Record<string, any>) => item.value === galleryImportForm.groupName),
)
const selectedGalleryFilterBatch = computed(() =>
  galleryBatchOptions.value.find((item: Record<string, any>) => item.value === galleryFilterForm.batchKey),
)
const selectedGalleryFilterLabel = computed(() =>
  galleryLabelOptions.value.find((item: Record<string, any>) => item.value === galleryFilterForm.labelName),
)
const visibleBatchKeys = computed(() =>
  galleryBatches.value.map((item: Record<string, any>) => `${item.source}||${item.splitName ?? ""}`),
)
const allVisibleBatchesSelected = computed(() =>
  visibleBatchKeys.value.length > 0 && visibleBatchKeys.value.every((key: string) => selectedGalleryBatchKeys.value.includes(key)),
)
const galleryUploadRowIds = computed(() => galleryUploadRows.value.map((item) => item.id))
const allGalleryUploadRowsSelected = computed(() =>
  galleryUploadRowIds.value.length > 0 && galleryUploadRowIds.value.every((id) => selectedGalleryUploadRowIds.value.includes(id)),
)
const datasetClassRows = computed(() => state.value.offline?.dataset?.classDistribution ?? state.value.offline?.dataset?.classes?.items ?? [])
const datasetDistributionMax = computed(() => {
  const values = datasetClassRows.value.flatMap((item: Record<string, unknown>) =>
    datasetSeries.map((series) => Number(item?.[series.key]) || 0),
  )
  return Math.max(1, ...values)
})
const trainingHistory = computed(() => state.value.offline?.training?.history?.items ?? [])
const evaluationHistory = computed(() => state.value.offline?.evaluation?.history ?? [])
const selectedEvaluationLatest = computed(() => {
  const history = evaluationHistory.value
  return history.length ? history[history.length - 1] : {}
})
const evaluationModels = computed(() => state.value.offline?.evaluation?.modelList?.items ?? [])
const retrievalGalleryItems = computed(() => state.value.retrieval?.gallery?.items ?? [])
const selectedRetrievalImage = computed(() =>
  retrievalGalleryItems.value.find((item: Record<string, unknown>) => Number(item.id) === selectedRetrievalImageId.value) ?? selectedRetrievalImageSnapshot.value,
)
const retrievalAttributeGroups = computed(() => state.value.retrieval?.attributeOptions ?? [])
const imageSearchResult = computed(() => state.value.retrieval?.imageResult)
const attributeSearchResult = computed(() => state.value.retrieval?.attributeResult)
const recognitionResult = computed(() => state.value.recognition?.result)
const recognitionFeature = computed(() => recognitionResult.value?.feature ?? {})
const classComparison = computed(() => recognitionResult.value?.classComparison ?? {})
const classComparisonRows = computed(() => classComparison.value?.classes ?? [])
const featureGroupRows = computed(() => classComparison.value?.featureGroups ?? [])
const visibleSearchResult = computed(() => (activeRetrievalTab.value === "attribute" ? attributeSearchResult.value : imageSearchResult.value))
const visibleResultPageTarget = computed(() => (activeRetrievalTab.value === "attribute" ? "attributeRetrievalResult" : "imageRetrievalResult"))
const visibleDuplicatePairs = computed(() => {
  const hidden = new Set(hiddenDuplicatePairKeys.value)
  return (state.value.duplicate?.pairs?.items ?? []).filter((item: Record<string, any>) => !hidden.has(duplicatePairKey(item.left.id, item.right.id)))
})
const duplicateThresholdEval = computed(() => state.value.duplicate?.thresholdEval ?? null)
const duplicateEvalRows = computed(() => duplicateThresholdEval.value?.rows ?? [])
const selectedAttributeLabels = computed(() => {
  const selected = new Set(selectedAttributeValues.value)
  return retrievalAttributeGroups.value
    .flatMap((group: Record<string, any>) => group.options ?? [])
    .filter((item: Record<string, any>) => selected.has(String(item.value)))
})
const attributeSearchModes = [
  { label: "特征相似度", value: "feature" },
  { label: "属性筛选", value: "filter" },
  { label: "属性 + 特征混合检索", value: "hybrid" },
]
const duplicateCurveSeries = [
  { key: "precision", label: "Precision", color: "#0f4c5c" },
  { key: "recall", label: "Recall", color: "#f2b95a" },
  { key: "f1", label: "F1", color: "#2a9d8f" },
]
const duplicateErrorSeries = [
  { key: "fp", label: "误检", color: "#c94c3d" },
  { key: "fn", label: "漏检", color: "#6b7fd7" },
]
const featureTypeOptions = [
  { label: "无", value: "none" },
  { label: "ResNet101特征嵌入", value: "resnet101" },
  { label: "自相似特征嵌入", value: "self_similarity" },
]
const resultSizeOptions = [10, 20, 50]
const offlineExperimentOptions = [
  {
    label: "特征效果对比",
    value: "feature_compare",
    description: "比较ResNet101特征嵌入和自相似特征嵌入，观察训练后特征是否提升mAP@K和Recall@K。",
    effect: "影响检索准确性",
    output: "输出两条结果，分别代表两种特征嵌入。",
  },
  {
    label: "索引方式对比",
    value: "index_compare",
    description: "固定一种特征，比较暴力检索、KD-Tree、HNSW和PQ的效果差异。",
    effect: "影响检索速度、索引大小和部分近似检索结果",
    output: "输出多条索引结果，用于比较速度、存储开销和准确性。",
  },
  {
    label: "重排序效果对比",
    value: "rerank_compare",
    description: "固定HNSW索引，比较是否启用候选重排序对结果质量和延时的影响。",
    effect: "影响Top K排序质量和检索延时",
    output: "输出HNSW初排和HNSW重排序两条结果。",
  },
]
const evaluationFeatureOptions = [
  {
    label: "ResNet101",
    value: "baseline",
  },
  {
    label: "自相似特征嵌入",
    value: "embedding",
  },
]

function pickFeatureManifest(options: string[], partition: "gallery" | "query", mode: "baseline" | "embedding") {
  return (
    options.find((item) => item.includes(`${partition}_${mode}`)) ??
    options.find((item) => item.includes(partition)) ??
    options[0] ??
    ""
  )
}

function experimentNameForGoal(goal: string) {
  if (goal === "feature_compare") return "exp2"
  if (goal === "rerank_compare") return "exp3"
  return "exp4"
}

function syncOfflineExperimentManifests(options: string[], force = false) {
  if (!options.length) return
  const baselineGallery = pickFeatureManifest(options, "gallery", "baseline")
  const baselineQuery = pickFeatureManifest(options, "query", "baseline")
  const embeddingGallery = pickFeatureManifest(options, "gallery", "embedding")
  const embeddingQuery = pickFeatureManifest(options, "query", "embedding")

  if (force || !options.includes(offlineExperimentForm.baselineGalleryManifest)) {
    offlineExperimentForm.baselineGalleryManifest = baselineGallery
  }
  if (force || !options.includes(offlineExperimentForm.baselineQueryManifest)) {
    offlineExperimentForm.baselineQueryManifest = baselineQuery
  }
  if (force || !options.includes(offlineExperimentForm.embeddingGalleryManifest)) {
    offlineExperimentForm.embeddingGalleryManifest = embeddingGallery
  }
  if (force || !options.includes(offlineExperimentForm.embeddingQueryManifest)) {
    offlineExperimentForm.embeddingQueryManifest = embeddingQuery
  }

  offlineExperimentForm.experimentName = "matrix"
  const useBaseline = offlineExperimentForm.featureScheme === "baseline"
  const selectedGallery = useBaseline ? baselineGallery : embeddingGallery
  const selectedQuery = useBaseline ? baselineQuery : embeddingQuery

  if (force || !options.includes(offlineExperimentForm.galleryManifest)) {
    offlineExperimentForm.galleryManifest = selectedGallery
  }
  if (force || !options.includes(offlineExperimentForm.queryManifest)) {
    offlineExperimentForm.queryManifest = selectedQuery
  }
}

watch(
  () => retrievalGalleryItems.value.map((item: Record<string, unknown>) => item.id).join(","),
  () => {
    const items = retrievalGalleryItems.value
    if (!items.length) {
      selectedRetrievalImageId.value = null
      selectedRetrievalImageSnapshot.value = null
      return
    }
    const currentItem = items.find((item: Record<string, unknown>) => Number(item.id) === selectedRetrievalImageId.value)
    if (!currentItem) {
      const firstItem = items[0] as Record<string, any>
      selectedRetrievalImageId.value = Number(firstItem.id)
      selectedRetrievalImageSnapshot.value = firstItem
      return
    }
    selectedRetrievalImageSnapshot.value = currentItem as Record<string, any>
  },
  { immediate: true },
)

const changeModule = (module: string) => {
  uiState.activeModule = module
}
const changeTab = (target: string, value: string) => {
  if (target === "gallery") {
    uiState.galleryTab = value
    return
  }
  if (target === "offline") {
    uiState.offlineTab = value
    if (value !== "evaluation") {
      evaluationDetailOpen.value = false
    }
    return
  }
  if (target === "retrieval") {
    uiState.retrievalTab = value
    return
  }
  if (target === "duplicate") {
    uiState.duplicateTab = value
    return
  }
  if (target === "cluster") {
    uiState.clusterTab = value
  }
}
const changePage = (target: string, page: number, totalPages?: number) => {
  const maxPage = Math.max(1, Number(totalPages) || Number.MAX_SAFE_INTEGER)
  const nextPage = Math.min(Math.max(1, Number(page) || 1), maxPage)
  sendEvent("setPage", { target, page: nextPage })
}
const jumpPage = (target: string, totalPages: number) => {
  const value = Number(pageJumpInputs[target])
  if (!Number.isFinite(value)) return
  changePage(target, value, totalPages)
}
const confirmAction = (message: string, action: () => void | Promise<void>) => {
  if (!window.confirm(message)) return
  return action()
}
const logout = () => confirmAction("确认退出登录？", () => sendEvent("logout"))

const submitLogin = () =>
  sendEvent("login", {
    username: loginForm.username.trim(),
    password: loginForm.password,
  })
const submitRegister = () =>
  sendEvent("register", {
    username: registerForm.username.trim(),
    displayName: registerForm.displayName.trim(),
    phone: registerForm.phone.trim(),
    email: registerForm.email.trim(),
    organization: registerForm.organization.trim(),
    password: registerForm.password,
    confirmPassword: registerForm.confirmPassword,
  })

const triggerAuthMode = (mode: "login" | "register") => {
  authMode.value = mode
  sendEvent("setAuthMode", { mode })
}

const onRetrievalUploadChange = (event: Event) => {
  const files = (event.target as HTMLInputElement).files
  retrievalUploadForm.file = files?.[0] ?? null
  retrievalUploadPreview.value = ""
  if (retrievalUploadForm.file) {
    retrievalUploadForm.imageUrl = ""
    fileToDataUrl(retrievalUploadForm.file).then((dataUrl) => {
      retrievalUploadPreview.value = dataUrl
    })
  }
}

const onGalleryUploadChange = (event: Event) => {
  const files = Array.from((event.target as HTMLInputElement).files ?? [])
  const rows = files.map((file, index) => ({
    id: `${file.name}-${file.size}-${file.lastModified}-${index}`,
    file,
    name: file.name,
    splitName: "",
    labelName: "__unset__",
  }))
  galleryUploadRows.value = rows
  selectedGalleryUploadRowIds.value = rows.map((item) => item.id)
}

const selectTestGroup = (value: string) => {
  galleryImportForm.groupName = value
  testGroupDropdownOpen.value = false
}

const selectGalleryFilterBatch = (value: string) => {
  galleryFilterForm.batchKey = value
  galleryFilterBatchDropdownOpen.value = false
}

const selectGalleryFilterLabel = (value: string) => {
  galleryFilterForm.labelName = value
  galleryFilterLabelDropdownOpen.value = false
}

const onRecognitionFileChange = (event: Event) => {
  const files = (event.target as HTMLInputElement).files
  recognitionForm.file = files?.[0] ?? null
  recognitionPreview.value = ""
  if (recognitionForm.file) {
    recognitionForm.imageUrl = ""
    fileToDataUrl(recognitionForm.file).then((dataUrl) => {
      recognitionPreview.value = dataUrl
    })
  }
}

const selectRetrievalImage = (imageId: number) => {
  selectedRetrievalImageId.value = Number(imageId)
  selectedRetrievalImageSnapshot.value = retrievalGalleryItems.value.find((item: Record<string, unknown>) => Number(item.id) === selectedRetrievalImageId.value) ?? null
}

const applyRetrievalGalleryFilters = () => {
  selectedRetrievalImageId.value = null
  selectedRetrievalImageSnapshot.value = null
  sendEvent("setRetrievalGalleryFilters", {
    batchKey: retrievalGalleryFilterForm.batchKey,
    labelName: retrievalGalleryFilterForm.labelName,
  })
}

const runGallerySearch = (imageId?: number | null) => {
  if (!imageId) return
  sendEvent("searchGallery", {
    imageId,
    method: retrievalGalleryForm.method,
    featureType: retrievalGalleryForm.featureType,
    rerankEnabled: retrievalGalleryForm.rerankEnabled,
    topK: retrievalGalleryForm.topK,
  })
}

const runUploadSearch = async () => {
  const imageUrl = retrievalUploadForm.imageUrl.trim()
  if (!retrievalUploadForm.file && !imageUrl) return
  if (imageUrl) {
    sendEvent("searchUploadUrl", {
      imageUrl,
      method: retrievalUploadForm.method,
      featureType: retrievalUploadForm.featureType,
      rerankEnabled: retrievalUploadForm.rerankEnabled,
      topK: retrievalUploadForm.topK,
    })
    return
  }
  if (!retrievalUploadForm.file) return
  sendEvent("searchUpload", {
    dataUrl: await fileToDataUrl(retrievalUploadForm.file),
    originalName: retrievalUploadForm.file.name,
    method: retrievalUploadForm.method,
    featureType: retrievalUploadForm.featureType,
    rerankEnabled: retrievalUploadForm.rerankEnabled,
    topK: retrievalUploadForm.topK,
  })
}

const importTestGroup = () =>
  confirmAction(`确认导入${galleryImportForm.groupName}？`, () => sendEvent("importTestGroup", {
    groupName: galleryImportForm.groupName,
    skipExisting: galleryImportForm.skipExisting,
  }))

const uploadGalleryImages = () =>
  confirmAction("确认上传图片？", async () => {
    const selectedIds = new Set(selectedGalleryUploadRowIds.value)
    const selectedRows = galleryUploadRows.value.filter((item) => selectedIds.has(item.id))
    if (!selectedRows.length) return
    const files = await Promise.all(
      selectedRows.map(async (item) => ({
        originalName: item.name,
        dataUrl: await fileToDataUrl(item.file),
        labelName: item.labelName,
        splitName: item.splitName,
      })),
    )
    sendEvent("uploadImages", {
      files,
    })
    galleryUploadRows.value = galleryUploadRows.value.filter((item) => !selectedIds.has(item.id))
    selectedGalleryUploadRowIds.value = []
  })

const toggleGalleryUploadRows = (event: Event) => {
  selectedGalleryUploadRowIds.value = (event.target as HTMLInputElement).checked ? galleryUploadRowIds.value.slice() : []
}

const deleteSelectedGalleryUploadRows = () => {
  const selected = new Set(selectedGalleryUploadRowIds.value)
  galleryUploadRows.value = galleryUploadRows.value.filter((item) => !selected.has(item.id))
  selectedGalleryUploadRowIds.value = []
}

const openCategoryDialog = () => {
  categoryDialogOpen.value = true
  categoryForm.id = null
  categoryForm.name = ""
}

const closeCategoryDialog = () => {
  categoryDialogOpen.value = false
  categoryForm.id = null
  categoryForm.name = ""
}

const editLabelCategory = (item: Record<string, any>) => {
  if (item.system) return
  categoryForm.id = Number(item.id)
  categoryForm.name = String(item.name ?? "")
}

const saveLabelCategory = () => {
  const name = categoryForm.name.trim()
  if (!name) return
  if (categoryForm.id) {
    confirmAction("确认保存分类？", () => sendEvent("updateLabelCategory", { id: categoryForm.id, name }))
  } else {
    confirmAction("确认新增分类？", () => sendEvent("createLabelCategory", { name }))
  }
  categoryForm.id = null
  categoryForm.name = ""
}

const deleteLabelCategory = (item: Record<string, any>) => {
  if (item.system) return
  confirmAction("确认删除分类？", () => sendEvent("deleteLabelCategory", { id: item.id }))
}

const toggleVisibleBatches = (event: Event) => {
  const checked = (event.target as HTMLInputElement).checked
  const selected = new Set(selectedGalleryBatchKeys.value)
  visibleBatchKeys.value.forEach((key: string) => {
    if (checked) {
      selected.add(key)
    } else {
      selected.delete(key)
    }
  })
  selectedGalleryBatchKeys.value = Array.from(selected)
}

const deleteSelectedGalleryBatches = () =>
  confirmAction("确认删除选择的批次？", () => {
    const selected = new Set(selectedGalleryBatchKeys.value)
    const batches = galleryBatchOptions.value
      .filter((item: Record<string, any>) => selected.has(item.value))
      .map((item: Record<string, any>) => ({
        source: item.source,
        splitName: item.splitName,
      }))
    if (!batches.length) return
    selectedGalleryBatchKeys.value = []
    sendEvent("deleteGalleryBatches", { batches })
  })

const applyGalleryFilters = () =>
  sendEvent("setGalleryFilters", {
    batchKey: galleryFilterForm.batchKey,
    labelName: galleryFilterForm.labelName,
  })

const resetGalleryFilters = () => {
  galleryFilterForm.batchKey = ""
  galleryFilterForm.labelName = ""
  applyGalleryFilters()
}

const toggleAttributeValue = (value: string) => {
  const next = new Set(selectedAttributeValues.value)
  if (next.has(value)) {
    next.delete(value)
  } else {
    next.add(value)
  }
  selectedAttributeValues.value = Array.from(next)
}

const runAttributeSearch = () => {
  if (!selectedAttributeValues.value.length) return
  sendEvent("searchAttributes", {
    attributes: selectedAttributeValues.value,
    topK: attributeSearchForm.topK,
    searchMode: attributeSearchForm.searchMode,
  })
}

const runAttributeRecognition = async () => {
  const imageUrl = recognitionForm.imageUrl.trim()
  if (!recognitionForm.file && !imageUrl) return
  if (imageUrl) {
    sendEvent("recognizeAttributes", { imageUrl })
    return
  }
  if (!recognitionForm.file) return
  sendEvent("recognizeAttributes", {
    dataUrl: await fileToDataUrl(recognitionForm.file),
    originalName: recognitionForm.file.name,
  })
}
const rebuildVectorIndex = () => confirmAction("确认更新向量索引？", () => sendEvent("rebuildVectorIndex"))
const refreshGalleryFeatures = () => confirmAction("确认刷新图库特征？", () => sendEvent("refreshGalleryFeatures"))

const runDuplicateScan = () => sendEvent("runDuplicate", { threshold: duplicateThreshold.value })
const runDuplicateThresholdEval = () =>
  sendEvent("runDuplicateThresholdEval", {
    topK: duplicateEvalTopK.value,
    sampleSize: duplicateEvalSampleSize.value,
  })
const useDuplicateThreshold = () => {
  duplicateThreshold.value = duplicatePresetValue.value
}
const runCluster = () => confirmAction("确认生成分组？", () => sendEvent("runCluster", { clusterCount: clusterCount.value }))
const openClusterDetail = (runId: number) => sendEvent("openClusterRun", { runId })
const closeClusterDetail = () => sendEvent("closeClusterRun")
const trainOfflineModel = () =>
  confirmAction("确认开始训练？", () => sendEvent("trainEmbeddingModel", {
    trainManifest: offlineTrainingForm.trainManifest,
    validationManifest: offlineTrainingForm.validationManifest,
    epochs: offlineTrainingForm.epochs,
    earlyStopPatience: offlineTrainingForm.earlyStopPatience,
    batchSize: offlineTrainingForm.batchSize,
    numWorkers: offlineTrainingForm.numWorkers,
    learningRate: offlineTrainingForm.learningRate,
    optimizerName: offlineTrainingForm.optimizerName,
    seed: offlineTrainingForm.seed,
    saveBestOnly: offlineTrainingForm.saveBestOnly,
    freezeBackbone: offlineTrainingForm.freezeBackbone,
    deviceName: offlineTrainingForm.deviceName,
  }))
const stopOfflineTraining = () => confirmAction("确认停止训练？", () => sendEvent("stopEmbeddingModel"))
const runOfflineExperiment = () => {
  syncOfflineExperimentManifests(featureManifestOptions.value, true)
  offlineExperimentForm.indexMethod = "all"
  offlineExperimentForm.historyKey = ""
  offlineExperimentStarting.value = true
  offlineExperimentStartAt.value = Date.now()
  offlineExperimentRuntimeOverride.value = {
    ...(offlineExperimentRuntime.value ?? {}),
    isRunning: true,
    status: "running",
    progress: 0,
  }
  void sendEvent("runOfflineExperiment", {
    experimentName: offlineExperimentForm.experimentName,
    featureScheme: offlineExperimentForm.featureScheme,
    indexMethod: "all",
    rerankEnabled: offlineExperimentForm.rerankEnabled,
    topK: offlineExperimentForm.topK,
    galleryManifest: offlineExperimentForm.galleryManifest,
    queryManifest: offlineExperimentForm.queryManifest,
    baselineGalleryManifest: offlineExperimentForm.baselineGalleryManifest,
    baselineQueryManifest: offlineExperimentForm.baselineQueryManifest,
    embeddingGalleryManifest: offlineExperimentForm.embeddingGalleryManifest,
    embeddingQueryManifest: offlineExperimentForm.embeddingQueryManifest,
  }).then((payload: any) => {
    const runtime = payload?.state?.offline?.experiments?.runtime
    if (payload?.state?.notice?.type === "error") {
      offlineExperimentStarting.value = false
      if (runtime) {
        offlineExperimentRuntimeOverride.value = runtime
      }
      return
    }
    if (runtime) {
      if (!runtime?.isRunning && runtime?.status !== "failed" && offlineExperimentStarting.value && Date.now() - offlineExperimentStartAt.value < 8000) {
        return
      }
      offlineExperimentStarting.value = false
      offlineExperimentRuntimeOverride.value = runtime
    }
  }).catch(() => {
    offlineExperimentStarting.value = false
    offlineExperimentRuntimeOverride.value = {
      ...(offlineExperimentRuntime.value ?? {}),
      isRunning: false,
      status: "failed",
    }
  })
}
const stopOfflineExperiment = () => confirmAction("确认停止评估？", () => {
  offlineExperimentStarting.value = false
  offlineExperimentRuntimeOverride.value = {
    ...(offlineExperimentRuntime.value ?? {}),
    isRunning: false,
    status: "cancelled",
  }
  sendEvent("stopOfflineExperiment")
})
const setEvaluationModel = (value: string) => sendEvent("setEvaluationModel", { value })
const openEvaluationDetail = (value: string) => {
  setEvaluationModel(value)
  evaluationDetailOpen.value = true
}
const closeEvaluationDetail = () => {
  evaluationDetailOpen.value = false
}
const deleteModelWeight = (value: string) => confirmAction("确认删除模型？", () => sendEvent("deleteModelWeight", { value }))
const removeImage = (imageId: number) => confirmAction("确认删除图片？", () => sendEvent("deleteImage", { imageId }))
const removeDuplicate = (keepId: number, deleteId: number, similarity: number) =>
  confirmAction("确认删除图片？", () => {
    hiddenDuplicatePairKeys.value = Array.from(new Set([...hiddenDuplicatePairKeys.value, duplicatePairKey(keepId, deleteId)]))
    duplicateRefreshing.value = true
    sendEvent("deleteDuplicate", {
      primaryImageId: keepId,
      duplicateImageId: deleteId,
      similarity,
      threshold: duplicateThreshold.value,
    })
    window.setTimeout(() => {
      duplicateRefreshing.value = false
    }, 1200)
  })

const updateProfile = () =>
  confirmAction("确认保存资料？", () => sendEvent("updateProfile", {
    displayName: profileForm.displayName,
    phone: profileForm.phone,
    email: profileForm.email,
    organization: profileForm.organization,
  }))

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

function numberValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "--"
  if (typeof value === "number") return value
  return String(value)
}

function metricNumber(value: unknown) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function offlineRecordLabel(item: Record<string, any>) {
  const parts = [item.featureLabel ?? item.displayName ?? item.name, item.indexLabel ?? item.indexType]
  if (item.rerank === "是") parts.push("重排序")
  return parts.filter(Boolean).join(" ")
}

function offlineMetricMax(key: string) {
  const values = filteredOfflineExperimentRecords.value
    .map((item: Record<string, any>) => metricNumber(item[key]))
    .filter((value): value is number => value !== null)
  if (key === "mapAtK" || key === "recallAtK") return Math.max(1, ...values)
  return Math.max(...values, 1)
}

function offlineMetricWidth(item: Record<string, any>, key: string) {
  const value = metricNumber(item[key])
  if (value === null || value <= 0) return "0%"
  const width = Math.max(3, Math.min(100, (value / offlineMetricMax(key)) * 100))
  return `${width.toFixed(2)}%`
}

function offlineMetricValue(item: Record<string, any>, key: string) {
  return metricNumber(item[key]) ?? 0
}

function formatBytes(value: unknown) {
  const numeric = metricNumber(value)
  if (numeric === null) return "--"
  if (numeric >= 1024 * 1024) return `${(numeric / 1024 / 1024).toFixed(2)}MB`
  if (numeric >= 1024) return `${(numeric / 1024).toFixed(2)}KB`
  return `${numeric.toFixed(0)}B`
}

function offlineMetricText(item: Record<string, any>, key: string, format: string) {
  const value = metricNumber(item[key])
  if (value === null) return "--"
  if (format === "percent") return `${(value * 100).toFixed(2)}%`
  if (format === "ms") return `${value.toFixed(2)}ms`
  if (format === "bytes") return formatBytes(value)
  return value.toFixed(2)
}

function offlineAxisText(value: number, format: string) {
  if (format === "percent") return `${(value * 100).toFixed(0)}%`
  if (format === "bytes") return formatBytes(value)
  if (format === "ms") return value >= 10 ? value.toFixed(0) : value.toFixed(2)
  return value.toFixed(2)
}

function offlineChartMaxFor(format: string, key: string) {
  const values = filteredOfflineExperimentRecords.value
    .map((item: Record<string, any>) => metricNumber(item[key]))
    .filter((value): value is number => value !== null)
  const maxValue = Math.max(...values, 0)
  if (format === "percent") return Math.max(1, maxValue)
  return maxValue > 0 ? maxValue * 1.12 : 1
}

function offlineChartTicks(format: string, key: string) {
  const maxValue = offlineChartMaxFor(format, key)
  return [maxValue, maxValue * 0.5, 0].map((value) => ({
    value,
    label: offlineAxisText(value, format),
    y: offlineBarY(value, maxValue),
  }))
}

function offlineBarWidth(total: number) {
  const plotWidth = chartFrame.width - chartFrame.left - chartFrame.right
  const segment = plotWidth / Math.max(total, 1)
  return Math.min(56, Math.max(26, segment * 0.54))
}

function offlineBarX(index: number, total: number) {
  const plotWidth = chartFrame.width - chartFrame.left - chartFrame.right
  const segment = plotWidth / Math.max(total, 1)
  const width = offlineBarWidth(total)
  return chartFrame.left + segment * index + (segment - width) / 2
}

function offlineBarY(value: number, maxValue: number) {
  const plotHeight = chartFrame.height - chartFrame.top - chartFrame.bottom
  const ratio = Math.max(0, Math.min(1, value / Math.max(maxValue, 1e-9)))
  return chartFrame.height - chartFrame.bottom - plotHeight * ratio
}

function offlineBarHeight(value: number, maxValue: number) {
  return chartFrame.height - chartFrame.bottom - offlineBarY(value, maxValue)
}

function duplicateChartX(index: number, total: number) {
  const plotWidth = chartFrame.width - chartFrame.left - chartFrame.right
  const segment = plotWidth / Math.max(total - 1, 1)
  return chartFrame.left + segment * index
}

function duplicateChartY(value: number, maxValue = 1) {
  const plotHeight = chartFrame.height - chartFrame.top - chartFrame.bottom
  const ratio = Math.max(0, Math.min(1, value / Math.max(maxValue, 1e-9)))
  return chartFrame.height - chartFrame.bottom - plotHeight * ratio
}

function duplicateCurvePoints(key: string) {
  return duplicateEvalRows.value
    .map((row: Record<string, any>, index: number) => `${duplicateChartX(index, duplicateEvalRows.value.length)},${duplicateChartY(metricNumber(row[key]) ?? 0)}`)
    .join(" ")
}

function duplicateErrorMax() {
  const values = duplicateEvalRows.value.flatMap((row: Record<string, any>) => duplicateErrorSeries.map((series) => metricNumber(row[series.key]) ?? 0))
  return Math.max(1, ...values)
}

function duplicateErrorBarWidth(total: number) {
  const plotWidth = chartFrame.width - chartFrame.left - chartFrame.right
  const segment = plotWidth / Math.max(total, 1)
  return Math.min(18, Math.max(8, segment * 0.24))
}

function duplicateErrorBarX(index: number, seriesIndex: number, total: number) {
  const plotWidth = chartFrame.width - chartFrame.left - chartFrame.right
  const segment = plotWidth / Math.max(total, 1)
  const width = duplicateErrorBarWidth(total)
  return chartFrame.left + segment * index + (segment - width * duplicateErrorSeries.length) / 2 + seriesIndex * width
}

function duplicateErrorBarHeight(value: number, maxValue: number) {
  return chartFrame.height - chartFrame.bottom - duplicateChartY(value, maxValue)
}

function duplicatePairKey(leftId: unknown, rightId: unknown) {
  return [Number(leftId), Number(rightId)].sort((left, right) => left - right).join("-")
}

function sourceText(value: unknown) {
  if (value === "test_set") return "测试集"
  if (value === "upload") return "上传"
  if (value === "cifar10_test") return "测试集"
  return numberValue(value)
}

function lossValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "--"
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return String(value)
  return numeric.toFixed(2)
}

function percentValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "--"
  const numeric = Number(value)
  if (Number.isNaN(numeric)) return String(value)
  return `${(numeric * 100).toFixed(2)}%`
}

function querySourceText(value: unknown) {
  if (value === "attributes") return "属性检索"
  if (value === "upload") return "图片检索"
  if (value === "url") return "图片链接"
  if (value === "gallery") return "图库图片"
  return numberValue(value)
}

function remotePreviewUrl(value: string) {
  const imageUrl = value.trim()
  if (!imageUrl) return ""
  return apiUrl(`/api/image-proxy?url=${encodeURIComponent(imageUrl)}`)
}

function decimalValue(value: unknown, digits = 6) {
  if (value === null || value === undefined || value === "") return "--"
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return String(value)
  return numeric.toFixed(digits)
}

function featureDifferenceWidth(value: unknown) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return "0%"
  return `${Math.max(8, Math.min(Math.abs(numeric) * 1200, 100))}%`
}

function barWidth(value: unknown, maxValue: number) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric <= 0) return "0%"
  return `${Math.max(1, (numeric / Math.max(maxValue, 1)) * 100).toFixed(2)}%`
}

function chartValues(records: Record<string, any>[], key: string) {
  return records
    .map((item) => {
      const value = Number(item?.[key])
      return Number.isFinite(value) ? value : null
    })
}

function chartHasData(records: Record<string, any>[], keys: string[]) {
  return keys.some((key) => chartValues(records, key).some((value) => value !== null))
}

function chartRange(records: Record<string, any>[], keys: string[]) {
  const numericValues = keys.flatMap((key) => chartValues(records, key)).filter((value): value is number => value !== null)
  if (!numericValues.length) {
    return { minValue: 0, maxValue: 1 }
  }
  const minValue = Math.min(...numericValues)
  const maxValue = Math.max(...numericValues)
  if (minValue === maxValue) {
    const padding = Math.abs(minValue) > 1 ? Math.abs(minValue) * 0.1 : 0.1
    return { minValue: minValue - padding, maxValue: maxValue + padding }
  }
  return {
    minValue,
    maxValue,
  }
}

function chartX(index: number, total: number) {
  const plotWidth = chartFrame.width - chartFrame.left - chartFrame.right
  if (total <= 1) return chartFrame.left
  return chartFrame.left + (plotWidth / (total - 1)) * index
}

function chartY(value: number, minValue: number, maxValue: number) {
  const plotHeight = chartFrame.height - chartFrame.top - chartFrame.bottom
  const span = maxValue - minValue || 1
  return chartFrame.height - chartFrame.bottom - ((value - minValue) / span) * plotHeight
}

function chartPath(records: Record<string, any>[], key: string, compareKeys: string[] = [key]) {
  const values = chartValues(records, key)
  const numericValues = values.filter((value): value is number => value !== null)
  if (!numericValues.length) return ""
  const { minValue, maxValue } = chartRange(records, compareKeys)
  return values
    .map((value, index) => {
      if (value === null) return null
      return `${chartX(index, records.length).toFixed(2)},${chartY(value, minValue, maxValue).toFixed(2)}`
    })
    .filter(Boolean)
    .join(" ")
}

function chartDots(records: Record<string, any>[], key: string, compareKeys: string[] = [key]) {
  const values = chartValues(records, key)
  const numericValues = values.filter((value): value is number => value !== null)
  if (!numericValues.length) return []
  const { minValue, maxValue } = chartRange(records, compareKeys)
  return values
    .map((value, index) => {
      if (value === null) return null
      return {
        x: chartX(index, records.length),
        y: chartY(value, minValue, maxValue),
        value,
      }
    })
    .filter((item): item is { x: number; y: number; value: number } => Boolean(item))
}

function chartYTicks(records: Record<string, any>[], keys: string[], format: "number" | "percent" = "number") {
  if (!chartHasData(records, keys)) return []
  const { minValue, maxValue } = chartRange(records, keys)
  const steps = 4
  return Array.from({ length: steps + 1 }, (_, index) => {
    const value = minValue + ((maxValue - minValue) / steps) * index
    return {
      value,
      y: chartY(value, minValue, maxValue),
      label: format === "percent" ? `${(value * 100).toFixed(0)}%` : formatChartNumber(value),
    }
  }).reverse()
}

function chartXTicks(records: Record<string, any>[]) {
  if (!records.length) return []
  const indexes = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => Math.round((records.length - 1) * ratio))
    .filter((index, position, values) => values.indexOf(index) === position)
  return indexes.map((index) => ({
    index,
    x: chartX(index, records.length),
    label: numberValue(records[index]?.epoch ?? index + 1),
  }))
}

function formatChartNumber(value: number) {
  if (Math.abs(value) >= 10) return value.toFixed(0)
  if (Math.abs(value) >= 1) return value.toFixed(2)
  return value.toFixed(2)
}

function matrixMax(matrix: number[][]) {
  const values = matrix.flat()
  return values.length ? Math.max(...values, 1) : 1
}

function heatColor(value: number, maxValue: number) {
  const ratio = Math.max(0, Math.min(1, value / Math.max(maxValue, 1)))
  const alpha = 0.14 + ratio * 0.82
  return `rgba(13, 59, 76, ${alpha.toFixed(3)})`
}
</script>

<template>
  <div class="page-root">
    <div v-if="toast" class="toast" :class="`toast-${toast.type}`">{{ toast.message }}</div>

    <section
      v-if="!state.authenticated"
      class="auth-screen"
      :style="{ backgroundImage: `linear-gradient(120deg, rgba(7,18,28,0.82), rgba(11,39,54,0.62)), url(${bgImage})` }"
    >
      <div class="auth-shell">
        <div class="auth-brand">
          <div class="brand-badge">IMG SEARCH</div>
          <h1>{{ state.appTitle }}</h1>
        </div>
        <div class="auth-card">
          <div class="auth-card-header">
            <h2>{{ authMode === "login" ? "账号登录" : "新用户注册" }}</h2>
            <p>{{ authMode === "login" ? "使用系统账号进入平台" : "填写基础信息后直接进入系统" }}</p>
          </div>
          <div class="auth-switch">
            <button type="button" :class="{ active: authMode === 'login' }" @click="triggerAuthMode('login')">登录</button>
            <button type="button" :class="{ active: authMode === 'register' }" @click="triggerAuthMode('register')">注册</button>
          </div>

          <div v-if="authMode === 'login'" class="form-grid">
            <label>
              <span>账号</span>
              <input v-model="loginForm.username" autocomplete="username" @keyup.enter="submitLogin" />
            </label>
            <label>
              <span>密码</span>
              <input v-model="loginForm.password" type="password" autocomplete="current-password" @keyup.enter="submitLogin" />
            </label>
            <div class="auth-action-stack">
              <button type="button" class="primary-btn" @click="submitLogin">进入系统</button>
              <button type="button" class="auth-link-btn" @click="triggerAuthMode('register')">没有账号？立即注册</button>
            </div>
          </div>

          <div v-else class="form-grid register-form-grid">
            <label class="form-row">
              <span>账号</span>
              <input v-model="registerForm.username" autocomplete="username" />
            </label>
            <label class="form-row">
              <span>姓名</span>
              <input v-model="registerForm.displayName" />
            </label>
            <label class="form-row">
              <span>手机号</span>
              <input v-model="registerForm.phone" autocomplete="tel" />
            </label>
            <label class="form-row">
              <span>邮箱</span>
              <input v-model="registerForm.email" autocomplete="email" />
            </label>
            <label class="form-row">
              <span>所属单位</span>
              <input v-model="registerForm.organization" />
            </label>
            <label class="form-row">
              <span>密码</span>
              <input v-model="registerForm.password" type="password" autocomplete="new-password" />
            </label>
            <label class="form-row">
              <span>确认密码</span>
              <input v-model="registerForm.confirmPassword" type="password" autocomplete="new-password" @keyup.enter="submitRegister" />
            </label>
            <div class="auth-action-stack">
              <button type="button" class="primary-btn" @click="submitRegister">注册并进入</button>
              <button type="button" class="auth-link-btn" @click="triggerAuthMode('login')">已有账号？返回登录</button>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section v-else class="app-shell">
      <aside class="sidebar">
        <div class="sidebar-top">
          <div class="sidebar-title">相似图片检索</div>
          <div class="sidebar-user">
            <strong>{{ state.user?.displayName }}</strong>
            <span>{{ state.user?.username }}</span>
          </div>
        </div>

        <nav class="menu">
          <button
            v-for="item in menuItems"
            :key="item.key"
            class="menu-item"
            :class="{ active: activeModule === item.key }"
            @click="changeModule(item.key)"
          >
            {{ item.label }}
          </button>
        </nav>

        <div class="sidebar-bottom">
          <button class="ghost-btn logout-btn" @click="logout">退出登录</button>
        </div>
      </aside>

      <main class="content">
        <section v-if="activeModule === 'gallery'" class="module-panel">
          <div class="tab-bar">
            <button :class="{ active: activeGalleryTab === 'batch' }" @click="changeTab('gallery', 'batch')">批量导入</button>
            <button :class="{ active: activeGalleryTab === 'upload' }" @click="changeTab('gallery', 'upload')">上传图片</button>
            <button :class="{ active: activeGalleryTab === 'list' }" @click="changeTab('gallery', 'list')">图库图片</button>
          </div>

          <div class="card-grid four">
            <article class="metric-card">
              <span>图库图片</span>
              <strong>{{ numberValue(state.gallery?.overview?.imageCount) }}</strong>
            </article>
            <article class="metric-card">
              <span>图库批次</span>
              <strong>{{ numberValue(state.gallery?.overview?.batchCount) }}</strong>
            </article>
            <article class="metric-card">
              <span>标签数量</span>
              <strong>{{ numberValue(state.gallery?.overview?.labelCount) }}</strong>
            </article>
            <article class="metric-card">
              <span>图库路径</span>
              <strong>{{ numberValue(state.gallery?.overview?.storageRoot) }}</strong>
            </article>
          </div>

          <div v-if="activeGalleryTab === 'batch'" class="panel-grid">
            <article class="toolbar-card">
              <div class="select-field">
                <span>测试集分组</span>
                <div class="custom-select">
                  <button class="custom-select-trigger" type="button" @click="testGroupDropdownOpen = !testGroupDropdownOpen">
                    {{ selectedTestGroup?.label ?? '请选择' }}（{{ selectedTestGroup?.imageCount ?? 0 }}张）
                  </button>
                  <div v-if="testGroupDropdownOpen" class="custom-select-menu">
                    <button
                      v-for="item in state.gallery?.testGroups ?? []"
                      :key="item.value"
                      type="button"
                      :class="{ selected: item.value === galleryImportForm.groupName }"
                      @click="selectTestGroup(item.value)"
                    >
                      {{ item.label }}（{{ item.imageCount }}张）
                    </button>
                  </div>
                </div>
              </div>
              <label class="checkbox-row">
                <input v-model="galleryImportForm.skipExisting" type="checkbox" />
                <span>跳过已存在图片</span>
              </label>
              <button class="primary-btn" @click="importTestGroup">导入图库</button>
            </article>

            <div class="table-card">
              <div class="table-card-header">
                <strong>已入库批次</strong>
                <button class="danger-btn" :disabled="!selectedGalleryBatchKeys.length" @click="deleteSelectedGalleryBatches">删除</button>
              </div>
              <table>
                <thead>
                  <tr>
                    <th class="check-col">
                      <input type="checkbox" :checked="allVisibleBatchesSelected" @change="toggleVisibleBatches" />
                    </th>
                    <th>批次</th>
                    <th>来源</th>
                    <th>数量</th>
                    <th>最近入库</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in galleryBatches" :key="`${item.source}-${item.splitName}`">
                    <td class="check-col">
                      <input
                        v-model="selectedGalleryBatchKeys"
                        type="checkbox"
                        :value="`${item.source}||${item.splitName ?? ''}`"
                      />
                    </td>
                    <td>{{ item.label }}</td>
                    <td>{{ sourceText(item.source) }}</td>
                    <td>{{ item.imageCount }}</td>
                    <td>{{ item.lastCreatedAt }}</td>
                  </tr>
                </tbody>
              </table>
              <div class="pager">
                <button :disabled="state.gallery?.batches?.page <= 1" @click="changePage('galleryBatches', 1, state.gallery?.batches?.totalPages)">
                  首页
                </button>
                <button :disabled="state.gallery?.batches?.page <= 1" @click="changePage('galleryBatches', state.gallery?.batches?.page - 1, state.gallery?.batches?.totalPages)">
                  上一页
                </button>
                <span>{{ state.gallery?.batches?.page }} / {{ state.gallery?.batches?.totalPages }}</span>
                <div class="pager-jump">
                  <input v-model="pageJumpInputs.galleryBatches" type="number" min="1" :max="state.gallery?.batches?.totalPages" />
                  <button @click="jumpPage('galleryBatches', state.gallery?.batches?.totalPages)">跳转</button>
                </div>
                <button :disabled="state.gallery?.batches?.page >= state.gallery?.batches?.totalPages" @click="changePage('galleryBatches', state.gallery?.batches?.page + 1, state.gallery?.batches?.totalPages)">
                  下一页
                </button>
                <button :disabled="state.gallery?.batches?.page >= state.gallery?.batches?.totalPages" @click="changePage('galleryBatches', state.gallery?.batches?.totalPages, state.gallery?.batches?.totalPages)">
                  末页
                </button>
              </div>
            </div>
          </div>

          <div v-if="activeGalleryTab === 'upload'" class="panel-grid">
            <article class="toolbar-card">
              <label>
                <span>图片文件</span>
                <input type="file" accept="image/*" multiple @change="onGalleryUploadChange" />
              </label>
            </article>

            <article class="table-card">
              <div class="table-card-header">
                <strong>待上传文件</strong>
                <div class="toolbar-actions">
                  <button class="danger-btn" :disabled="!selectedGalleryUploadRowIds.length" @click="deleteSelectedGalleryUploadRows">删除</button>
                  <button class="primary-btn" :disabled="!selectedGalleryUploadRowIds.length" @click="uploadGalleryImages">入库</button>
                </div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th class="check-col">
                      <input type="checkbox" :checked="allGalleryUploadRowsSelected" @change="toggleGalleryUploadRows" />
                    </th>
                    <th>文件名</th>
                    <th>批次</th>
                    <th>标签类别</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in galleryUploadRows" :key="item.id">
                    <td class="check-col">
                      <input v-model="selectedGalleryUploadRowIds" type="checkbox" :value="item.id" />
                    </td>
                    <td class="upload-file-name">{{ item.name }}</td>
                    <td>
                      <select v-model="item.splitName" class="table-select">
                        <option v-for="option in galleryUploadBatchOptions" :key="option.value" :value="option.value">
                          {{ option.label }}
                        </option>
                      </select>
                    </td>
                    <td>
                      <select v-model="item.labelName" class="table-select">
                        <option v-for="option in galleryLabelOptions" :key="option.value" :value="option.value">
                          {{ option.label }}
                        </option>
                      </select>
                    </td>
                  </tr>
                </tbody>
              </table>
            </article>
          </div>

          <div v-if="activeGalleryTab === 'list'" class="panel-grid">
            <article class="toolbar-card">
              <div class="select-field">
                <span>批次</span>
                <div class="custom-select">
                  <button class="custom-select-trigger" type="button" @click="galleryFilterBatchDropdownOpen = !galleryFilterBatchDropdownOpen">
                    {{ selectedGalleryFilterBatch?.label ?? '全部批次' }}
                  </button>
                  <div v-if="galleryFilterBatchDropdownOpen" class="custom-select-menu">
                    <button type="button" :class="{ selected: galleryFilterForm.batchKey === '' }" @click="selectGalleryFilterBatch('')">
                      全部批次
                    </button>
                    <button
                      v-for="item in galleryBatchOptions"
                      :key="item.value"
                      type="button"
                      :class="{ selected: galleryFilterForm.batchKey === item.value }"
                      @click="selectGalleryFilterBatch(item.value)"
                    >
                      {{ item.label }}
                    </button>
                  </div>
                </div>
              </div>
              <div class="select-field">
                <span>标签类别</span>
                <div class="custom-select">
                  <button class="custom-select-trigger" type="button" @click="galleryFilterLabelDropdownOpen = !galleryFilterLabelDropdownOpen">
                    {{ selectedGalleryFilterLabel?.label ?? '全部标签' }}
                  </button>
                  <div v-if="galleryFilterLabelDropdownOpen" class="custom-select-menu">
                    <button type="button" :class="{ selected: galleryFilterForm.labelName === '' }" @click="selectGalleryFilterLabel('')">
                      全部标签
                    </button>
                    <button
                      v-for="item in galleryLabelOptions"
                      :key="item.value"
                      type="button"
                      :class="{ selected: galleryFilterForm.labelName === item.value }"
                      @click="selectGalleryFilterLabel(item.value)"
                    >
                      {{ item.label }}
                    </button>
                  </div>
                </div>
              </div>
              <button class="primary-btn" @click="applyGalleryFilters">筛选</button>
              <button class="ghost-btn" @click="resetGalleryFilters">重置</button>
              <button class="ghost-btn" @click="openCategoryDialog">分类管理</button>
            </article>
            <div class="image-grid">
              <article v-for="item in galleryImages" :key="item.id" class="image-card">
                <img :src="item.thumbnail" :alt="item.originalName" />
                <div class="image-meta">
                  <strong>{{ item.originalName }}</strong>
                  <span>{{ item.labelName || '未设置' }}</span>
                  <span>{{ item.source }} {{ item.splitName || '' }}</span>
                </div>
                <button class="danger-btn" @click="removeImage(item.id)">删除</button>
              </article>
            </div>
            <div class="pager">
              <button :disabled="state.gallery?.images?.page <= 1" @click="changePage('gallery', 1, state.gallery?.images?.totalPages)">
                首页
              </button>
              <button :disabled="state.gallery?.images?.page <= 1" @click="changePage('gallery', state.gallery?.images?.page - 1, state.gallery?.images?.totalPages)">
                上一页
              </button>
              <span>{{ state.gallery?.images?.page }} / {{ state.gallery?.images?.totalPages }}</span>
              <div class="pager-jump">
                <input v-model="pageJumpInputs.gallery" type="number" min="1" :max="state.gallery?.images?.totalPages" />
                <button @click="jumpPage('gallery', state.gallery?.images?.totalPages)">跳转</button>
              </div>
              <button :disabled="state.gallery?.images?.page >= state.gallery?.images?.totalPages" @click="changePage('gallery', state.gallery?.images?.page + 1, state.gallery?.images?.totalPages)">
                下一页
              </button>
              <button :disabled="state.gallery?.images?.page >= state.gallery?.images?.totalPages" @click="changePage('gallery', state.gallery?.images?.totalPages, state.gallery?.images?.totalPages)">
                末页
              </button>
            </div>
          </div>

          <div v-if="categoryDialogOpen" class="dialog-backdrop" @click.self="closeCategoryDialog">
            <section class="dialog-panel category-dialog">
              <header class="dialog-header">
                <strong>分类管理</strong>
                <button class="ghost-btn" @click="closeCategoryDialog">关闭</button>
              </header>
              <div class="dialog-body">
                <article class="toolbar-card">
                  <label>
                    <span>分类名称</span>
                    <input v-model="categoryForm.name" />
                  </label>
                  <button class="primary-btn" @click="saveLabelCategory">{{ categoryForm.id ? '保存' : '新增' }}</button>
                  <button class="ghost-btn" @click="categoryForm.id = null; categoryForm.name = ''">清空</button>
                </article>
                <article class="table-card">
                  <table>
                    <thead>
                      <tr>
                        <th>分类名称</th>
                        <th>图片数量</th>
                        <th>类型</th>
                        <th>操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="item in galleryLabelCategories" :key="item.id">
                        <td>{{ item.name }}</td>
                        <td>{{ item.imageCount }}</td>
                        <td>{{ item.system ? '系统分类' : '自定义分类' }}</td>
                        <td>
                          <div class="table-actions">
                            <button class="ghost-btn" :disabled="item.system" @click="editLabelCategory(item)">编辑</button>
                            <button class="danger-btn" :disabled="item.system" @click="deleteLabelCategory(item)">删除</button>
                          </div>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </article>
              </div>
            </section>
          </div>
        </section>

        <section v-if="activeModule === 'offline'" class="module-panel">
          <div class="tab-bar">
            <button :class="{ active: activeOfflineTab === 'dataset' }" @click="changeTab('offline', 'dataset')">训练数据集</button>
            <button :class="{ active: activeOfflineTab === 'training' }" @click="changeTab('offline', 'training')">模型训练</button>
            <button :class="{ active: activeOfflineTab === 'evaluation' }" @click="changeTab('offline', 'evaluation')">模型评估</button>
          </div>

          <div v-if="activeOfflineTab === 'dataset'" class="panel-grid">
            <div class="section-title">数据规模</div>
            <div class="card-grid four">
              <article class="metric-card">
                <span>数据集</span>
                <strong>CIFAR10</strong>
              </article>
              <article class="metric-card">
                <span>训练集</span>
                <strong>{{ numberValue(state.offline?.dataset?.summary?.preparedTrainCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>测试集</span>
                <strong>{{ numberValue(state.offline?.dataset?.summary?.preparedTestCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>类别数</span>
                <strong>{{ numberValue(state.offline?.dataset?.summary?.classCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>图库</span>
                <strong>{{ numberValue(state.offline?.dataset?.summary?.preparedGalleryCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>查询集</span>
                <strong>{{ numberValue(state.offline?.dataset?.summary?.preparedQueryCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>有效样本</span>
                <strong>{{ numberValue(state.offline?.dataset?.summary?.validCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>重复数量</span>
                <strong>{{ numberValue(state.offline?.dataset?.summary?.duplicateCount) }}</strong>
              </article>
            </div>

            <div class="section-title">类别分布</div>
            <article class="distribution-card">
              <div class="distribution-legend">
                <span v-for="series in datasetSeries" :key="series.key">
                  <i :class="series.className"></i>{{ series.label }}
                </span>
              </div>
              <div class="distribution-list">
                <div v-for="item in datasetClassRows" :key="item.labelName" class="distribution-row">
                  <strong>{{ item.labelName }}</strong>
                  <div class="distribution-bars">
                    <div v-for="series in datasetSeries" :key="`${item.labelName}-${series.key}`" class="distribution-bar-line">
                      <span>{{ series.label }}</span>
                      <div class="distribution-track">
                        <div class="distribution-bar" :class="series.className" :style="{ width: barWidth(item[series.key], datasetDistributionMax) }"></div>
                      </div>
                      <em>{{ numberValue(item[series.key]) }}</em>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </div>

          <div v-if="activeOfflineTab === 'training'" class="panel-grid">
            <article class="toolbar-card">
              <label>
                <span>训练清单</span>
                <select v-model="offlineTrainingForm.trainManifest">
                  <option v-for="item in state.offline?.training?.manifestOptions ?? []" :key="item" :value="item">
                    {{ item }}
                  </option>
                </select>
              </label>
              <label>
                <span>验证清单</span>
                <select v-model="offlineTrainingForm.validationManifest">
                  <option v-for="item in state.offline?.training?.validationManifestOptions ?? []" :key="`val-${item}`" :value="item">
                    {{ item }}
                  </option>
                </select>
              </label>
              <label>
                <span>Epoch</span>
                <input v-model.number="offlineTrainingForm.epochs" type="number" min="1" max="500" />
              </label>
              <label>
                <span>Early Stop</span>
                <input v-model.number="offlineTrainingForm.earlyStopPatience" type="number" min="0" max="100" />
              </label>
              <label>
                <span>Batch</span>
                <input v-model.number="offlineTrainingForm.batchSize" type="number" min="1" max="256" />
              </label>
              <label>
                <span>Workers</span>
                <input v-model.number="offlineTrainingForm.numWorkers" type="number" min="0" max="16" />
              </label>
              <label>
                <span>学习率</span>
                <input v-model.number="offlineTrainingForm.learningRate" type="number" min="0.00001" max="1" step="0.0001" />
              </label>
              <label>
                <span>优化器</span>
                <select v-model="offlineTrainingForm.optimizerName">
                  <option v-for="item in state.offline?.training?.optimizerOptions ?? []" :key="item.value" :value="item.value">
                    {{ item.label }}
                  </option>
                </select>
              </label>
              <label>
                <span>Seed</span>
                <input v-model.number="offlineTrainingForm.seed" type="number" min="0" max="999999" />
              </label>
              <label>
                <span>微调策略</span>
                <select v-model="offlineTrainingForm.freezeBackbone">
                  <option :value="true">冻结骨干</option>
                  <option :value="false">全量微调</option>
                </select>
              </label>
              <label>
                <span>最佳保存</span>
                <select v-model="offlineTrainingForm.saveBestOnly">
                  <option :value="true">仅最佳</option>
                  <option :value="false">全部保留 latest</option>
                </select>
              </label>
              <label>
                <span>设备</span>
                <select v-model="offlineTrainingForm.deviceName">
                  <option
                    v-for="item in state.offline?.training?.deviceOptions ?? []"
                    :key="item.value"
                    :value="item.value"
                    :disabled="item.available === false"
                  >
                    {{ item.label }}
                  </option>
                </select>
              </label>
              <div class="toolbar-actions">
                <button class="primary-btn" :disabled="trainingIsRunning" @click="trainOfflineModel">
                  {{ trainingIsRunning ? '训练中...' : '开始训练' }}
                </button>
                <button class="ghost-btn" :disabled="!trainingIsRunning" @click="stopOfflineTraining">
                  {{ trainingRuntimeStatus === 'stopping' ? '停止中...' : '停止训练' }}
                </button>
              </div>
            </article>

            <div class="section-title">训练过程</div>
            <div class="table-card training-process-card">
              <table>
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>状态</th>
                    <th>训练 Loss</th>
                    <th>训练 Acc</th>
                    <th>训练 Precision</th>
                    <th>训练 Recall</th>
                    <th>训练 F1</th>
                    <th>验证 Loss</th>
                    <th>验证 Acc</th>
                    <th>验证 Precision</th>
                    <th>验证 Recall</th>
                    <th>验证 F1</th>
                    <th>样本数</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in trainingHistory" :key="item.epoch">
                    <td>{{ item.epoch }}</td>
                    <td>{{ item.status ?? '--' }}</td>
                    <td>{{ lossValue(item.loss) }}</td>
                    <td>{{ percentValue(item.accuracy) }}</td>
                    <td>{{ percentValue(item.precision) }}</td>
                    <td>{{ percentValue(item.recall) }}</td>
                    <td>{{ percentValue(item.macroF1) }}</td>
                    <td>{{ lossValue(item.valLoss) }}</td>
                    <td>{{ percentValue(item.valAccuracy) }}</td>
                    <td>{{ percentValue(item.valPrecision) }}</td>
                    <td>{{ percentValue(item.valRecall) }}</td>
                    <td>{{ percentValue(item.valMacroF1) }}</td>
                    <td>{{ numberValue(item.sampleCount) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="pager">
              <button
                :disabled="state.offline?.training?.history?.page <= 1"
                @click="changePage('offlineTraining', state.offline?.training?.history?.page - 1)"
              >
                上一页
              </button>
              <span>{{ state.offline?.training?.history?.page }} / {{ state.offline?.training?.history?.totalPages }}</span>
              <button
                :disabled="state.offline?.training?.history?.page >= state.offline?.training?.history?.totalPages"
                @click="changePage('offlineTraining', state.offline?.training?.history?.page + 1)"
              >
                下一页
              </button>
            </div>

          </div>

          <div v-if="activeOfflineTab === 'evaluation'" class="panel-grid">
            <div class="section-title">模型列表</div>
            <div class="table-card model-list-card">
              <table>
                <thead>
                  <tr>
                    <th>模型名称</th>
                    <th>验证 Accuracy</th>
                    <th>验证 F1</th>
                    <th>验证 Loss</th>
                    <th>最佳 Epoch</th>
                    <th>状态</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in evaluationModels" :key="item.value" :class="{ selected: item.value === state.offline?.evaluation?.selectedModel }">
                    <td>
                      <button class="table-link-btn" @click="openEvaluationDetail(item.value)">{{ item.modelName }}</button>
                    </td>
                    <td>{{ percentValue(item.valAccuracy) }}</td>
                    <td>{{ percentValue(item.valMacroF1) }}</td>
                    <td>{{ lossValue(item.valLoss) }}</td>
                    <td>{{ numberValue(item.bestEpoch) }}</td>
                    <td>{{ numberValue(item.status) }}</td>
                    <td>
                      <div class="table-actions">
                        <button class="ghost-btn" @click="openEvaluationDetail(item.value)">详情</button>
                        <button class="danger-btn" :disabled="!item.deletable" @click="deleteModelWeight(item.value)">删除</button>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="pager">
              <button
                :disabled="state.offline?.evaluation?.modelList?.page <= 1"
                @click="changePage('offlineModels', state.offline?.evaluation?.modelList?.page - 1)"
              >
                上一页
              </button>
              <span>{{ state.offline?.evaluation?.modelList?.page }} / {{ state.offline?.evaluation?.modelList?.totalPages }}</span>
              <button
                :disabled="state.offline?.evaluation?.modelList?.page >= state.offline?.evaluation?.modelList?.totalPages"
                @click="changePage('offlineModels', state.offline?.evaluation?.modelList?.page + 1)"
              >
                下一页
              </button>
            </div>

            <div v-if="evaluationDetailOpen" class="dialog-backdrop" @click.self="closeEvaluationDetail">
              <section class="dialog-panel model-detail-dialog">
                <header class="dialog-header">
                  <strong>模型详情</strong>
                  <button class="ghost-btn" @click="closeEvaluationDetail">关闭</button>
                </header>
                <div class="dialog-body">
            <div class="section-title">训练结果</div>
            <div class="card-grid four">
              <article class="metric-card">
                <span>训练 Loss</span>
                <strong>{{ lossValue(selectedEvaluationLatest.loss) }}</strong>
              </article>
              <article class="metric-card">
                <span>训练 Accuracy</span>
                <strong>{{ percentValue(selectedEvaluationLatest.accuracy) }}</strong>
              </article>
              <article class="metric-card">
                <span>训练 F1</span>
                <strong>{{ percentValue(selectedEvaluationLatest.macroF1) }}</strong>
              </article>
              <article class="metric-card">
                <span>最佳 Epoch</span>
                <strong>{{ numberValue(state.offline?.evaluation?.summary?.bestEpoch) }}</strong>
              </article>
              <article class="metric-card">
                <span>验证 Loss</span>
                <strong>{{ lossValue(state.offline?.evaluation?.summary?.latestValLoss) }}</strong>
              </article>
              <article class="metric-card">
                <span>验证 Accuracy</span>
                <strong>{{ percentValue(state.offline?.evaluation?.summary?.latestValAccuracy) }}</strong>
              </article>
              <article class="metric-card">
                <span>验证 F1</span>
                <strong>{{ percentValue(state.offline?.evaluation?.summary?.latestValMacroF1) }}</strong>
              </article>
              <article class="metric-card">
                <span>类别数</span>
                <strong>{{ numberValue(state.offline?.evaluation?.summary?.classCount) }}</strong>
              </article>
            </div>

            <div class="section-title">模型内容</div>
            <div class="card-grid three detail-grid">
              <article v-for="item in state.offline?.training?.modelInfo ?? []" :key="`detail-model-${item.label}`" class="detail-card">
                <span>{{ item.label }}</span>
                <strong>{{ numberValue(item.value) }}</strong>
              </article>
            </div>
            <div class="model-architecture-flow">
              <article v-for="item in state.offline?.training?.modelArchitecture ?? []" :key="`flow-${item.stage}`" class="model-architecture-step">
                <span>{{ item.stage }}</span>
                <strong>{{ item.module }}</strong>
                <em>{{ item.output }}</em>
              </article>
            </div>
            <div class="table-card">
              <table>
                <thead>
                  <tr>
                    <th>层级</th>
                    <th>模块</th>
                    <th>输出</th>
                    <th>训练状态</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in state.offline?.training?.modelArchitecture ?? []" :key="`arch-${item.stage}`">
                    <td>{{ item.stage }}</td>
                    <td>{{ item.module }}</td>
                    <td>{{ item.output }}</td>
                    <td>{{ item.trainable }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="section-title">训练记录</div>
            <div class="card-grid three detail-grid">
              <article v-for="item in state.offline?.evaluation?.runInfo ?? []" :key="`evaluation-run-${item.label}`" class="detail-card">
                <span>{{ item.label }}</span>
                <strong>{{ numberValue(item.value) }}</strong>
              </article>
            </div>

            <div class="section-title">训练曲线</div>

            <div class="chart-grid">
              <article class="chart-card">
                <header>
                  <strong>Loss 曲线</strong>
                </header>
                <template v-if="chartHasData(evaluationHistory, ['loss', 'valLoss'])">
                  <svg class="chart-svg" viewBox="0 0 540 220">
                    <g v-for="(tick, index) in chartYTicks(evaluationHistory, ['loss', 'valLoss'])" :key="`eval-loss-y-${index}`">
                      <line :x1="chartFrame.left" :x2="chartFrame.width - chartFrame.right" :y1="tick.y" :y2="tick.y" class="chart-grid-line" />
                      <text :x="chartFrame.left - 8" :y="tick.y + 4" text-anchor="end" class="chart-y-label">{{ tick.label }}</text>
                    </g>
                    <g v-for="tick in chartXTicks(evaluationHistory)" :key="`eval-loss-x-${tick.index}`">
                      <line :x1="tick.x" :x2="tick.x" :y1="chartFrame.height - chartFrame.bottom" :y2="chartFrame.height - chartFrame.bottom + 5" class="chart-axis-tick" />
                      <text :x="tick.x" :y="chartFrame.height - chartFrame.bottom + 22" text-anchor="middle" class="chart-x-label">{{ tick.label }}</text>
                    </g>
                    <line :x1="chartFrame.left" :y1="chartFrame.height - chartFrame.bottom" :x2="chartFrame.width - chartFrame.right" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                    <line :x1="chartFrame.left" :y1="chartFrame.top" :x2="chartFrame.left" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                    <polyline :points="chartPath(evaluationHistory, 'loss', ['loss', 'valLoss'])" class="chart-line chart-line-train" />
                    <polyline :points="chartPath(evaluationHistory, 'valLoss', ['loss', 'valLoss'])" class="chart-line chart-line-val" />
                    <circle v-for="(dot, index) in chartDots(evaluationHistory, 'loss', ['loss', 'valLoss'])" :key="`loss-train-${index}`" :cx="dot.x" :cy="dot.y" r="3.5" class="chart-dot chart-dot-train" />
                    <circle v-for="(dot, index) in chartDots(evaluationHistory, 'valLoss', ['loss', 'valLoss'])" :key="`loss-val-${index}`" :cx="dot.x" :cy="dot.y" r="3.5" class="chart-dot chart-dot-val" />
                  </svg>
                  <div class="chart-legend">
                    <span><i class="legend-swatch legend-train"></i>训练</span>
                    <span><i class="legend-swatch legend-val"></i>验证</span>
                  </div>
                </template>
                <div v-else class="empty-card">暂无结果</div>
              </article>

              <article class="chart-card">
                <header>
                  <strong>Accuracy 趋势</strong>
                </header>
                <template v-if="chartHasData(evaluationHistory, ['accuracy', 'valAccuracy'])">
                  <svg class="chart-svg" viewBox="0 0 540 220">
                    <g v-for="(tick, index) in chartYTicks(evaluationHistory, ['accuracy', 'valAccuracy'], 'percent')" :key="`eval-acc-y-${index}`">
                      <line :x1="chartFrame.left" :x2="chartFrame.width - chartFrame.right" :y1="tick.y" :y2="tick.y" class="chart-grid-line" />
                      <text :x="chartFrame.left - 8" :y="tick.y + 4" text-anchor="end" class="chart-y-label">{{ tick.label }}</text>
                    </g>
                    <g v-for="tick in chartXTicks(evaluationHistory)" :key="`eval-acc-x-${tick.index}`">
                      <line :x1="tick.x" :x2="tick.x" :y1="chartFrame.height - chartFrame.bottom" :y2="chartFrame.height - chartFrame.bottom + 5" class="chart-axis-tick" />
                      <text :x="tick.x" :y="chartFrame.height - chartFrame.bottom + 22" text-anchor="middle" class="chart-x-label">{{ tick.label }}</text>
                    </g>
                    <line :x1="chartFrame.left" :y1="chartFrame.height - chartFrame.bottom" :x2="chartFrame.width - chartFrame.right" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                    <line :x1="chartFrame.left" :y1="chartFrame.top" :x2="chartFrame.left" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                    <polyline :points="chartPath(evaluationHistory, 'accuracy', ['accuracy', 'valAccuracy'])" class="chart-line chart-line-train" />
                    <polyline :points="chartPath(evaluationHistory, 'valAccuracy', ['accuracy', 'valAccuracy'])" class="chart-line chart-line-val" />
                    <circle v-for="(dot, index) in chartDots(evaluationHistory, 'accuracy', ['accuracy', 'valAccuracy'])" :key="`acc-train-${index}`" :cx="dot.x" :cy="dot.y" r="3.5" class="chart-dot chart-dot-train" />
                    <circle v-for="(dot, index) in chartDots(evaluationHistory, 'valAccuracy', ['accuracy', 'valAccuracy'])" :key="`acc-val-${index}`" :cx="dot.x" :cy="dot.y" r="3.5" class="chart-dot chart-dot-val" />
                  </svg>
                  <div class="chart-legend">
                    <span><i class="legend-swatch legend-train"></i>训练</span>
                    <span><i class="legend-swatch legend-val"></i>验证</span>
                  </div>
                </template>
                <div v-else class="empty-card">暂无结果</div>
              </article>

              <article class="chart-card">
                <header>
                  <strong>F1 趋势</strong>
                </header>
                <template v-if="chartHasData(evaluationHistory, ['macroF1', 'valMacroF1'])">
                  <svg class="chart-svg" viewBox="0 0 540 220">
                    <g v-for="(tick, index) in chartYTicks(evaluationHistory, ['macroF1', 'valMacroF1'], 'percent')" :key="`eval-f1-y-${index}`">
                      <line :x1="chartFrame.left" :x2="chartFrame.width - chartFrame.right" :y1="tick.y" :y2="tick.y" class="chart-grid-line" />
                      <text :x="chartFrame.left - 8" :y="tick.y + 4" text-anchor="end" class="chart-y-label">{{ tick.label }}</text>
                    </g>
                    <g v-for="tick in chartXTicks(evaluationHistory)" :key="`eval-f1-x-${tick.index}`">
                      <line :x1="tick.x" :x2="tick.x" :y1="chartFrame.height - chartFrame.bottom" :y2="chartFrame.height - chartFrame.bottom + 5" class="chart-axis-tick" />
                      <text :x="tick.x" :y="chartFrame.height - chartFrame.bottom + 22" text-anchor="middle" class="chart-x-label">{{ tick.label }}</text>
                    </g>
                    <line :x1="chartFrame.left" :y1="chartFrame.height - chartFrame.bottom" :x2="chartFrame.width - chartFrame.right" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                    <line :x1="chartFrame.left" :y1="chartFrame.top" :x2="chartFrame.left" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                    <polyline :points="chartPath(evaluationHistory, 'macroF1', ['macroF1', 'valMacroF1'])" class="chart-line chart-line-train" />
                    <polyline :points="chartPath(evaluationHistory, 'valMacroF1', ['macroF1', 'valMacroF1'])" class="chart-line chart-line-val" />
                    <circle v-for="(dot, index) in chartDots(evaluationHistory, 'macroF1', ['macroF1', 'valMacroF1'])" :key="`f1-train-${index}`" :cx="dot.x" :cy="dot.y" r="3.5" class="chart-dot chart-dot-train" />
                    <circle v-for="(dot, index) in chartDots(evaluationHistory, 'valMacroF1', ['macroF1', 'valMacroF1'])" :key="`f1-val-${index}`" :cx="dot.x" :cy="dot.y" r="3.5" class="chart-dot chart-dot-val" />
                  </svg>
                  <div class="chart-legend">
                    <span><i class="legend-swatch legend-train"></i>训练</span>
                    <span><i class="legend-swatch legend-val"></i>验证</span>
                  </div>
                </template>
                <div v-else class="empty-card">暂无结果</div>
              </article>
            </div>

            <div class="section-title">混淆矩阵</div>
            <article v-if="state.offline?.evaluation?.confusionMatrix?.length" class="matrix-card">
              <div class="matrix-header matrix-grid" :style="{ gridTemplateColumns: `140px repeat(${state.offline?.evaluation?.classNames?.length ?? 0}, minmax(54px, 1fr))` }">
                <div class="matrix-corner">真实\\预测</div>
                <div v-for="label in state.offline?.evaluation?.classNames ?? []" :key="`matrix-col-${label}`" class="matrix-label">{{ label }}</div>
              </div>
              <div
                v-for="(row, rowIndex) in state.offline?.evaluation?.confusionMatrix ?? []"
                :key="`matrix-row-${rowIndex}`"
                class="matrix-grid"
                :style="{ gridTemplateColumns: `140px repeat(${state.offline?.evaluation?.classNames?.length ?? 0}, minmax(54px, 1fr))` }"
              >
                <div class="matrix-label">{{ state.offline?.evaluation?.classNames?.[rowIndex] ?? rowIndex }}</div>
                <div
                  v-for="(value, columnIndex) in row"
                  :key="`matrix-cell-${rowIndex}-${columnIndex}`"
                  class="matrix-cell"
                  :style="{ background: heatColor(value, matrixMax(state.offline?.evaluation?.confusionMatrix ?? [])) }"
                >
                  {{ value }}
                </div>
              </div>
            </article>
            <div v-else class="empty-card">暂无结果</div>

            <div class="section-title">各类别识别表现</div>
            <div class="table-card">
              <table>
                <thead>
                  <tr>
                    <th>类别</th>
                    <th>样本数</th>
                    <th>查准率</th>
                    <th>查全率</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in state.offline?.evaluation?.perClassMetrics?.items ?? []" :key="item.labelIndex">
                    <td>{{ item.labelName }}</td>
                    <td>{{ item.support }}</td>
                    <td>{{ percentValue(item.precision) }}</td>
                    <td>{{ percentValue(item.recall) }}</td>
                    <td>{{ percentValue(item.f1) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="pager">
              <button
                :disabled="state.offline?.evaluation?.perClassMetrics?.page <= 1"
                @click="changePage('offlineEvaluation', state.offline?.evaluation?.perClassMetrics?.page - 1)"
              >
                上一页
              </button>
              <span>{{ state.offline?.evaluation?.perClassMetrics?.page }} / {{ state.offline?.evaluation?.perClassMetrics?.totalPages }}</span>
              <button
                :disabled="state.offline?.evaluation?.perClassMetrics?.page >= state.offline?.evaluation?.perClassMetrics?.totalPages"
                @click="changePage('offlineEvaluation', state.offline?.evaluation?.perClassMetrics?.page + 1)"
              >
                下一页
              </button>
            </div>
                </div>
              </section>
            </div>
          </div>

        </section>

        <section v-if="activeModule === 'recognition'" class="module-panel">
          <div class="recognition-workspace">
            <aside class="search-panel recognition-panel">
              <div class="search-panel-title">特征识别</div>
              <label class="query-uploader">
                <input type="file" accept="image/*" @change="onRecognitionFileChange" />
                <img
                  v-if="recognitionPreview || recognitionForm.imageUrl"
                  :src="recognitionPreview || remotePreviewUrl(recognitionForm.imageUrl)"
                  alt="识别图片"
                />
                <span v-else>识别图片</span>
              </label>
              <label>
                <span>图片链接</span>
                <input v-model.trim="recognitionForm.imageUrl" type="url" placeholder="https://..." @input="recognitionPreview = ''; recognitionForm.file = null" />
              </label>
              <button class="primary-btn" :disabled="!recognitionForm.file && !recognitionForm.imageUrl.trim()" @click="runAttributeRecognition">开始识别</button>
            </aside>

            <section class="search-result-panel recognition-result-panel">
              <template v-if="recognitionResult">
                <header class="recognition-header">
                  <div class="query-summary-card">
                    <img :src="recognitionResult.query.thumbnail" :alt="recognitionResult.query.name" />
                    <div class="query-meta">
                      <strong>{{ recognitionResult.query.name }}</strong>
                      <span>{{ querySourceText(recognitionResult.query.source) }}</span>
                    </div>
                  </div>
                  <div class="result-count-card">
                    <span>相似类别</span>
                    <strong>{{ recognitionResult.predictedLabel }}</strong>
                  </div>
                  <div class="result-count-card">
                    <span>类别置信度</span>
                    <strong>{{ percentValue(recognitionResult.confidence) }}</strong>
                  </div>
                </header>

                <div class="recognition-section">
                  <strong>模型特征</strong>
                  <div class="feature-stat-grid">
                    <div class="candidate-card">
                      <span>模型</span>
                      <strong>{{ recognitionFeature.model }}</strong>
                    </div>
                    <div class="candidate-card">
                      <span>特征类型</span>
                      <strong>{{ recognitionFeature.featureType }}</strong>
                    </div>
                    <div class="candidate-card">
                      <span>维度</span>
                      <strong>{{ recognitionFeature.dimension }}</strong>
                    </div>
                    <div class="candidate-card">
                      <span>图库样本</span>
                      <strong>{{ classComparison.totalImages }}</strong>
                    </div>
                  </div>
                </div>

                <div class="recognition-section">
                  <strong>类别匹配对比</strong>
                  <div class="class-prototype-grid">
                    <div v-for="item in classComparisonRows" :key="item.label" class="class-prototype-card">
                      <header>
                        <strong>{{ item.label }}</strong>
                        <span>{{ item.sampleCount }}张</span>
                      </header>
                      <div class="prototype-score">
                        <span>综合匹配度</span>
                        <strong>{{ percentValue(item.decisionScore) }}</strong>
                      </div>
                      <div class="prototype-mini-grid">
                        <small>原型相似度 {{ percentValue(item.prototypeSimilarity) }}</small>
                        <small>最近样本 {{ percentValue(item.nearestSimilarity) }}</small>
                        <small>Top5均值 {{ percentValue(item.topAverageSimilarity) }}</small>
                        <small>Top20命中 {{ item.topEvidenceCount }}</small>
                      </div>
                    </div>
                  </div>
                </div>

                <div class="recognition-section">
                  <strong>类别区别</strong>
                  <div class="feature-stat-grid">
                    <div class="candidate-card">
                      <span>最接近类别原型</span>
                      <strong>{{ classComparison.primaryLabel }}</strong>
                    </div>
                    <div class="candidate-card">
                      <span>第二接近类别</span>
                      <strong>{{ classComparison.runnerUpLabel }}</strong>
                    </div>
                    <div class="candidate-card">
                      <span>综合匹配度间隔</span>
                      <strong>{{ percentValue(classComparison.decisionMargin) }}</strong>
                    </div>
                    <div class="candidate-card">
                      <span>最近样本间隔</span>
                      <strong>{{ percentValue(classComparison.nearestMargin) }}</strong>
                    </div>
                  </div>
                </div>

                <div class="recognition-section">
                  <strong>关键区别特征组</strong>
                  <div class="feature-group-list">
                    <div v-for="item in featureGroupRows" :key="item.name" class="feature-group-row">
                      <div>
                        <span>{{ item.name }}</span>
                        <strong>支持 {{ item.supportLabel }}</strong>
                      </div>
                      <div class="feature-group-track">
                        <i
                          :class="{ negative: Number(item.difference) < 0 }"
                          :style="{ width: featureDifferenceWidth(item.difference) }"
                        ></i>
                      </div>
                      <em>{{ decimalValue(item.difference, 4) }}</em>
                    </div>
                  </div>
                </div>

              </template>
              <div v-else class="empty-card">暂无结果</div>
            </section>
          </div>
        </section>

        <section v-if="activeModule === 'retrieval'" class="module-panel">
          <div class="tab-bar">
            <button :class="{ active: activeRetrievalTab === 'image' }" @click="changeTab('retrieval', 'image')">图片检索</button>
            <button :class="{ active: activeRetrievalTab === 'evaluation' }" @click="changeTab('retrieval', 'evaluation')">离线评估</button>
          </div>

          <div v-if="activeRetrievalTab === 'image'" class="search-workspace">
            <aside v-if="activeRetrievalTab === 'image'" class="search-panel">
              <div class="search-panel-title">查询图片</div>
              <div class="segmented-control">
                <button :class="{ selected: imageQueryMode === 'upload' }" @click="imageQueryMode = 'upload'">上传图片</button>
                <button :class="{ selected: imageQueryMode === 'gallery' }" @click="imageQueryMode = 'gallery'">图库图片</button>
              </div>
              <template v-if="imageQueryMode === 'upload'">
                <label class="query-uploader">
                  <input type="file" accept="image/*" @change="onRetrievalUploadChange" />
                  <img
                    v-if="retrievalUploadPreview || retrievalUploadForm.imageUrl"
                    :src="retrievalUploadPreview || remotePreviewUrl(retrievalUploadForm.imageUrl)"
                    alt="查询图片"
                  />
                  <span v-else>查询图片</span>
                </label>
                <label>
                  <span>图片链接</span>
                  <input v-model.trim="retrievalUploadForm.imageUrl" type="url" placeholder="https://..." @input="retrievalUploadPreview = ''; retrievalUploadForm.file = null" />
                </label>
                <button class="primary-btn" :disabled="!retrievalUploadForm.file && !retrievalUploadForm.imageUrl.trim()" @click="runUploadSearch">开始检索</button>
              </template>
              <template v-else>
                <div class="retrieval-filter-row">
                  <label>
                    <span>批次</span>
                    <select v-model="retrievalGalleryFilterForm.batchKey" @change="applyRetrievalGalleryFilters">
                      <option value="">全部批次</option>
                      <option v-for="item in galleryBatchOptions" :key="item.value" :value="item.value">
                        {{ item.label }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>标签类别</span>
                    <select v-model="retrievalGalleryFilterForm.labelName" @change="applyRetrievalGalleryFilters">
                      <option value="">全部标签</option>
                      <option v-for="item in galleryLabelOptions" :key="item.value" :value="item.value">
                        {{ item.label }}
                      </option>
                    </select>
                  </label>
                </div>
                <div class="gallery-query-grid">
                  <button
                    v-for="item in retrievalGalleryItems"
                    :key="item.id"
                    type="button"
                    :class="{ selected: item.id === selectedRetrievalImageId }"
                    @click="selectRetrievalImage(item.id)"
                  >
                    <img :src="item.thumbnail" :alt="item.originalName" />
                  </button>
                </div>
                <div class="compact-pager">
                  <button :disabled="state.retrieval?.gallery?.page <= 1" @click="changePage('retrievalGallery', state.retrieval?.gallery?.page - 1)">上一页</button>
                  <span>{{ state.retrieval?.gallery?.page }} / {{ state.retrieval?.gallery?.totalPages }}</span>
                  <button :disabled="state.retrieval?.gallery?.page >= state.retrieval?.gallery?.totalPages" @click="changePage('retrievalGallery', state.retrieval?.gallery?.page + 1)">下一页</button>
                </div>
                <button class="primary-btn" :disabled="!selectedRetrievalImageId" @click="runGallerySearch(selectedRetrievalImageId)">开始检索</button>
              </template>
              <div class="attribute-group">
                <strong>特征嵌入</strong>
                <select v-model="retrievalUploadForm.featureType" @change="retrievalGalleryForm.featureType = retrievalUploadForm.featureType">
                  <option v-for="item in featureTypeOptions" :key="item.value" :value="item.value">
                    {{ item.label }}
                  </option>
                </select>
              </div>
              <div class="attribute-group">
                <strong>索引方式</strong>
                <select v-model="retrievalUploadForm.method" @change="retrievalGalleryForm.method = retrievalUploadForm.method">
                  <option v-for="item in state.retrieval?.methods ?? []" :key="item.value" :value="item.value">
                    {{ item.label }}
                  </option>
                </select>
              </div>
              <label class="checkbox-row">
                <input v-model="retrievalUploadForm.rerankEnabled" type="checkbox" @change="retrievalGalleryForm.rerankEnabled = retrievalUploadForm.rerankEnabled" />
                <span>重排序</span>
              </label>
              <div class="attribute-group">
                <strong>返回结果数</strong>
                <div class="attribute-chip-list">
                  <button
                    v-for="value in resultSizeOptions"
                    :key="`image-topk-${value}`"
                    type="button"
                    class="attribute-chip"
                    :class="{ selected: retrievalUploadForm.topK === value }"
                    @click="retrievalUploadForm.topK = value; retrievalGalleryForm.topK = value"
                  >
                    {{ value }}
                  </button>
                </div>
              </div>
            </aside>

            <aside v-if="activeRetrievalTab === 'attribute'" class="search-panel attribute-search-panel">
              <div class="search-panel-title">属性检索</div>
              <div class="attribute-group">
                <strong>检索方式</strong>
                <div class="attribute-chip-list">
                  <button
                    v-for="item in attributeSearchModes"
                    :key="item.value"
                    type="button"
                    class="attribute-chip wide"
                    :class="{ selected: attributeSearchForm.searchMode === item.value }"
                    @click="attributeSearchForm.searchMode = item.value"
                  >
                    {{ item.label }}
                  </button>
                </div>
              </div>
              <div class="attribute-group">
                <strong>返回结果数</strong>
                <div class="attribute-chip-list">
                  <button
                    v-for="value in resultSizeOptions"
                    :key="`attr-topk-${value}`"
                    type="button"
                    class="attribute-chip"
                    :class="{ selected: attributeSearchForm.topK === value }"
                    @click="attributeSearchForm.topK = value"
                  >
                    {{ value }}
                  </button>
                </div>
              </div>
              <div class="attribute-group" v-for="group in retrievalAttributeGroups" :key="group.key">
                <strong>{{ group.label }}</strong>
                <div class="attribute-chip-list">
                  <button
                    v-for="item in group.options"
                    :key="item.value"
                    type="button"
                    class="attribute-chip"
                    :class="{ selected: selectedAttributeValues.includes(item.value) }"
                    @click="toggleAttributeValue(item.value)"
                  >
                    {{ item.label }}
                  </button>
                </div>
              </div>
              <div v-if="selectedAttributeLabels.length" class="selected-attribute-row">
                <span v-for="item in selectedAttributeLabels" :key="item.value">{{ item.label }}</span>
              </div>
              <button class="primary-btn" :disabled="!selectedAttributeValues.length" @click="runAttributeSearch">按属性检索</button>
            </aside>

            <section class="search-result-panel">
              <header class="search-result-header">
                <div v-if="visibleSearchResult" class="query-summary-card">
                  <img
                    v-if="visibleSearchResult.query.thumbnail"
                    :src="visibleSearchResult.query.thumbnail"
                    :alt="visibleSearchResult.query.name"
                  />
                  <div v-else class="attribute-query-icon">属性</div>
                  <div class="query-meta">
                    <strong>{{ visibleSearchResult.query.name }}</strong>
                    <div v-if="visibleSearchResult.query.attributes?.length" class="result-tag-row">
                      <span v-for="item in visibleSearchResult.query.attributes" :key="item.value">{{ item.label }}</span>
                    </div>
                    <span v-else>{{ querySourceText(visibleSearchResult.query.source) }}</span>
                  </div>
                </div>
                <div class="result-count-card">
                  <span>检索结果</span>
                  <strong>{{ numberValue(visibleSearchResult?.list?.total ?? 0) }}</strong>
                </div>
              </header>

              <template v-if="visibleSearchResult">
                <div class="search-results-grid">
                  <article v-for="item in visibleSearchResult.list.items" :key="item.id" class="search-result-card">
                    <img :src="item.thumbnail" :alt="item.originalName" />
                    <div class="search-result-meta">
                      <strong>{{ item.originalName }}</strong>
                      <em>匹配度 {{ percentValue(item.score) }}</em>
                      <div class="result-tag-row">
                        <span v-for="tag in item.attributes ?? []" :key="`${item.id}-${tag.value}`">{{ tag.label }}</span>
                      </div>
                    </div>
                  </article>
                </div>
                <div class="pager">
                  <button
                    :disabled="visibleSearchResult.list.page <= 1"
                    @click="changePage(visibleResultPageTarget, visibleSearchResult.list.page - 1)"
                  >
                    上一页
                  </button>
                  <span>{{ visibleSearchResult.list.page }} / {{ visibleSearchResult.list.totalPages }}</span>
                  <button
                    :disabled="visibleSearchResult.list.page >= visibleSearchResult.list.totalPages"
                    @click="changePage(visibleResultPageTarget, visibleSearchResult.list.page + 1)"
                  >
                    下一页
                  </button>
                </div>
              </template>
              <div v-else class="empty-card">暂无结果</div>
            </section>
          </div>

          <div v-if="activeRetrievalTab === 'evaluation'" class="panel-grid">
            <article class="toolbar-card evaluation-toolbar">
              <label class="evaluation-static-field">
                <span>实验库图片</span>
                <strong>{{ numberValue(offlineEvaluationGalleryCount) }}</strong>
              </label>
              <label>
                <span>特征嵌入</span>
                <select v-model="offlineExperimentForm.featureScheme">
                  <option v-for="item in evaluationFeatureOptions" :key="item.value" :value="item.value">
                    {{ item.label }}
                  </option>
                </select>
              </label>
              <label class="checkbox-row evaluation-checkbox">
                <input v-model="offlineExperimentForm.rerankEnabled" type="checkbox" />
                <span>重排序</span>
              </label>
              <button class="primary-btn" :disabled="offlineExperimentBusy || trainingIsRunning" @click="runOfflineExperiment">运行评估</button>
              <button v-if="offlineExperimentBusy" class="danger-btn" @click="stopOfflineExperiment">停止评估</button>
              <div class="evaluation-progress">
                <div class="evaluation-progress-track">
                  <span :style="{ width: `${offlineExperimentProgress}%` }"></span>
                </div>
                <strong>{{ Math.round(offlineExperimentProgress) }}%</strong>
                <small v-if="offlineExperimentStatusText" class="evaluation-status">{{ offlineExperimentStatusText }}</small>
              </div>
            </article>
            <article class="toolbar-card evaluation-history-toolbar">
              <label>
                <span>历史记录</span>
                <select v-model="offlineExperimentForm.historyKey">
                  <option v-for="item in offlineExperimentHistoryOptions" :key="item.value" :value="item.value">
                    {{ item.label }}
                  </option>
                </select>
              </label>
            </article>

            <section class="evaluation-chart-grid">
              <article v-for="chart in offlineEvaluationCharts" :key="chart.key" class="evaluation-chart-card">
                <header>{{ chart.title }}</header>
                <svg v-if="filteredOfflineExperimentRecords.length" class="evaluation-bar-svg" :viewBox="`0 0 ${chartFrame.width} ${chartFrame.height}`" role="img">
                  <g v-for="tick in offlineChartTicks(chart.format, chart.key)" :key="`${chart.key}-tick-${tick.label}`">
                    <line :x1="chartFrame.left" :x2="chartFrame.width - chartFrame.right" :y1="tick.y" :y2="tick.y" class="chart-grid-line" />
                    <text :x="chartFrame.left - 10" :y="tick.y + 4" text-anchor="end" class="chart-y-label">{{ tick.label }}</text>
                  </g>
                  <line :x1="chartFrame.left" :y1="chartFrame.height - chartFrame.bottom" :x2="chartFrame.width - chartFrame.right" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                  <line :x1="chartFrame.left" :y1="chartFrame.top" :x2="chartFrame.left" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                  <g v-for="(item, index) in filteredOfflineExperimentRecords" :key="`${chart.key}-${item.name}`">
                    <rect
                      :x="offlineBarX(index, filteredOfflineExperimentRecords.length)"
                      :y="offlineBarY(offlineMetricValue(item, chart.key), offlineChartMaxFor(chart.format, chart.key))"
                      :width="offlineBarWidth(filteredOfflineExperimentRecords.length)"
                      :height="offlineBarHeight(offlineMetricValue(item, chart.key), offlineChartMaxFor(chart.format, chart.key))"
                      :fill="chart.color"
                      rx="5"
                    />
                    <text
                      :x="offlineBarX(index, filteredOfflineExperimentRecords.length) + offlineBarWidth(filteredOfflineExperimentRecords.length) / 2"
                      :y="offlineBarY(offlineMetricValue(item, chart.key), offlineChartMaxFor(chart.format, chart.key)) - 6"
                      text-anchor="middle"
                      class="chart-x-label"
                    >
                      {{ offlineMetricText(item, chart.key, chart.format) }}
                    </text>
                    <text
                      :x="offlineBarX(index, filteredOfflineExperimentRecords.length) + offlineBarWidth(filteredOfflineExperimentRecords.length) / 2"
                      :y="chartFrame.height - chartFrame.bottom + 22"
                      text-anchor="middle"
                      class="chart-x-label"
                    >
                      {{ item.indexLabel ?? item.indexType }}
                    </text>
                  </g>
                </svg>
                <div v-else class="empty-state">暂无结果</div>
              </article>
            </section>

            <div class="table-card">
              <table>
                <thead>
                  <tr>
                    <th>特征嵌入</th>
                    <th>模型</th>
                    <th>索引方式</th>
                    <th>实验库</th>
                    <th>mAP@K</th>
                    <th>Recall@K</th>
                    <th>Precision@K</th>
                    <th>检索延时(ms)</th>
                    <th>索引大小</th>
                    <th>重排序</th>
                    <th>更新时间</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in visibleOfflineExperimentRecords" :key="item.name">
                    <td>{{ item.featureLabel ?? item.displayName ?? item.name }}</td>
                    <td>{{ item.featureModelName ?? "--" }}</td>
                    <td>{{ item.indexLabel ?? item.indexType }}</td>
                    <td>{{ item.galleryCount }}</td>
                    <td>{{ offlineMetricText(item, "mapAtK", "percent") }}</td>
                    <td>{{ offlineMetricText(item, "recallAtK", "percent") }}</td>
                    <td>{{ offlineMetricText(item, "precisionAtK", "percent") }}</td>
                    <td>{{ offlineMetricText(item, "elapsedMs", "ms") }}</td>
                    <td>{{ formatBytes(item.indexSizeBytes) }}</td>
                    <td>{{ item.rerank }}</td>
                    <td>{{ item.updatedAt }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="pager">
              <button
                :disabled="offlineExperimentLocalPage <= 1"
                @click="offlineExperimentLocalPage = Math.max(1, offlineExperimentLocalPage - 1)"
              >
                上一页
              </button>
              <span>{{ offlineExperimentLocalPage }} / {{ offlineExperimentTotalPages }}</span>
              <button
                :disabled="offlineExperimentLocalPage >= offlineExperimentTotalPages"
                @click="offlineExperimentLocalPage = Math.min(offlineExperimentTotalPages, offlineExperimentLocalPage + 1)"
              >
                下一页
              </button>
            </div>
          </div>

        </section>

        <section v-if="activeModule === 'duplicate'" class="module-panel">
          <div class="tab-bar">
            <button type="button" :class="{ active: activeDuplicateTab === 'scan' }" @click="changeTab('duplicate', 'scan')">重复扫描</button>
            <button type="button" :class="{ active: activeDuplicateTab === 'threshold' }" @click="changeTab('duplicate', 'threshold')">阈值优化</button>
          </div>

          <template v-if="activeDuplicateTab === 'scan'">
            <article class="toolbar-card">
              <label>
                <span>阈值方案</span>
                <select v-model.number="duplicatePresetValue" @change="useDuplicateThreshold">
                  <option v-for="item in duplicateThresholdPresets" :key="item.value" :value="item.value">
                    {{ item.label }}
                  </option>
                </select>
              </label>
              <label>
                <span>相似度阈值</span>
                <input v-model.number="duplicateThreshold" type="number" min="0.5" max="0.999" step="0.001" />
              </label>
              <button class="primary-btn" @click="runDuplicateScan">开始扫描</button>
              <div v-if="duplicateRefreshing" class="inline-status">正在重新分析...</div>
            </article>

            <div class="duplicate-list">
              <article v-for="item in visibleDuplicatePairs" :key="`${item.left.id}-${item.right.id}`" class="duplicate-card">
                <div class="duplicate-image">
                  <img :src="item.left.thumbnail" :alt="item.left.name" />
                  <strong>{{ item.left.name }}</strong>
                  <button class="danger-btn" @click="removeDuplicate(item.right.id, item.left.id, item.similarity)">删除</button>
                </div>
                <div class="duplicate-score">{{ percentValue(item.similarity) }}</div>
                <div class="duplicate-image">
                  <img :src="item.right.thumbnail" :alt="item.right.name" />
                  <strong>{{ item.right.name }}</strong>
                  <button class="danger-btn" @click="removeDuplicate(item.left.id, item.right.id, item.similarity)">删除</button>
                </div>
              </article>
            </div>
            <div class="pager">
              <button
                :disabled="state.duplicate?.pairs?.page <= 1"
                @click="changePage('duplicate', state.duplicate?.pairs?.page - 1)"
              >
                上一页
              </button>
              <span>{{ state.duplicate?.pairs?.page }} / {{ state.duplicate?.pairs?.totalPages }}</span>
              <button
                :disabled="state.duplicate?.pairs?.page >= state.duplicate?.pairs?.totalPages"
                @click="changePage('duplicate', state.duplicate?.pairs?.page + 1)"
              >
                下一页
              </button>
            </div>
          </template>

          <template v-else>
            <article class="toolbar-card">
              <label>
                <span>TopK</span>
                <input v-model.number="duplicateEvalTopK" type="number" min="1" max="50" />
              </label>
              <label>
                <span>评估样本</span>
                <input v-model.number="duplicateEvalSampleSize" type="number" min="10" max="500" step="10" />
              </label>
              <button class="primary-btn" @click="runDuplicateThresholdEval">运行评估</button>
            </article>

            <div v-if="duplicateThresholdEval" class="card-grid four">
              <article class="metric-card">
                <span>图库图片</span>
                <strong>{{ numberValue(duplicateThresholdEval.galleryCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>重复对</span>
                <strong>{{ numberValue(duplicateThresholdEval.positivePairCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>非重复对</span>
                <strong>{{ numberValue(duplicateThresholdEval.negativePairCount) }}</strong>
              </article>
              <article class="metric-card">
                <span>推荐阈值</span>
                <strong>{{ decimalValue(duplicateThresholdEval.recommendedThreshold, 3) }}</strong>
              </article>
            </div>

            <section v-if="duplicateEvalRows.length" class="evaluation-chart-grid">
              <article class="evaluation-chart-card">
                <header>Precision Recall F1</header>
                <svg class="evaluation-bar-svg" :viewBox="`0 0 ${chartFrame.width} ${chartFrame.height}`" role="img">
                  <g v-for="tick in [1, 0.5, 0]" :key="`dup-metric-${tick}`">
                    <line :x1="chartFrame.left" :x2="chartFrame.width - chartFrame.right" :y1="duplicateChartY(tick)" :y2="duplicateChartY(tick)" class="chart-grid-line" />
                    <text :x="chartFrame.left - 10" :y="duplicateChartY(tick) + 4" text-anchor="end" class="chart-y-label">{{ percentValue(tick) }}</text>
                  </g>
                  <line :x1="chartFrame.left" :y1="chartFrame.height - chartFrame.bottom" :x2="chartFrame.width - chartFrame.right" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                  <line :x1="chartFrame.left" :y1="chartFrame.top" :x2="chartFrame.left" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                  <polyline
                    v-for="series in duplicateCurveSeries"
                    :key="series.key"
                    :points="duplicateCurvePoints(series.key)"
                    :stroke="series.color"
                    fill="none"
                    stroke-width="3"
                  />
                  <template v-for="(row, index) in duplicateEvalRows" :key="`dup-label-${row.threshold}`">
                    <text :x="duplicateChartX(index, duplicateEvalRows.length)" :y="chartFrame.height - chartFrame.bottom + 22" text-anchor="middle" class="chart-x-label">
                      {{ decimalValue(row.threshold, 2) }}
                    </text>
                  </template>
                </svg>
                <div class="result-tag-row">
                  <span v-for="series in duplicateCurveSeries" :key="series.key">{{ series.label }}</span>
                </div>
              </article>
              <article class="evaluation-chart-card">
                <header>误检 漏检</header>
                <svg class="evaluation-bar-svg" :viewBox="`0 0 ${chartFrame.width} ${chartFrame.height}`" role="img">
                  <g v-for="tick in [duplicateErrorMax(), duplicateErrorMax() * 0.5, 0]" :key="`dup-error-${tick}`">
                    <line :x1="chartFrame.left" :x2="chartFrame.width - chartFrame.right" :y1="duplicateChartY(tick, duplicateErrorMax())" :y2="duplicateChartY(tick, duplicateErrorMax())" class="chart-grid-line" />
                    <text :x="chartFrame.left - 10" :y="duplicateChartY(tick, duplicateErrorMax()) + 4" text-anchor="end" class="chart-y-label">{{ Math.round(tick) }}</text>
                  </g>
                  <line :x1="chartFrame.left" :y1="chartFrame.height - chartFrame.bottom" :x2="chartFrame.width - chartFrame.right" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                  <line :x1="chartFrame.left" :y1="chartFrame.top" :x2="chartFrame.left" :y2="chartFrame.height - chartFrame.bottom" class="chart-axis" />
                  <template v-for="(row, index) in duplicateEvalRows" :key="`dup-error-row-${row.threshold}`">
                    <rect
                      v-for="(series, seriesIndex) in duplicateErrorSeries"
                      :key="`${row.threshold}-${series.key}`"
                      :x="duplicateErrorBarX(index, seriesIndex, duplicateEvalRows.length)"
                      :y="duplicateChartY(metricNumber(row[series.key]) ?? 0, duplicateErrorMax())"
                      :width="duplicateErrorBarWidth(duplicateEvalRows.length)"
                      :height="duplicateErrorBarHeight(metricNumber(row[series.key]) ?? 0, duplicateErrorMax())"
                      :fill="series.color"
                      rx="4"
                    />
                    <text :x="duplicateChartX(index, duplicateEvalRows.length)" :y="chartFrame.height - chartFrame.bottom + 22" text-anchor="middle" class="chart-x-label">
                      {{ decimalValue(row.threshold, 2) }}
                    </text>
                  </template>
                </svg>
                <div class="result-tag-row">
                  <span v-for="series in duplicateErrorSeries" :key="series.key">{{ series.label }}</span>
                </div>
              </article>
            </section>

            <article v-if="duplicateThresholdEval?.rows?.length" class="table-card">
              <table>
                <thead>
                  <tr>
                    <th>阈值</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>TP</th>
                    <th>FP</th>
                    <th>FN</th>
                    <th>TN</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in duplicateThresholdEval.rows" :key="row.threshold">
                    <td>{{ decimalValue(row.threshold, 3) }}</td>
                    <td>{{ percentValue(row.precision) }}</td>
                    <td>{{ percentValue(row.recall) }}</td>
                    <td>{{ percentValue(row.f1) }}</td>
                    <td>{{ numberValue(row.tp) }}</td>
                    <td>{{ numberValue(row.fp) }}</td>
                    <td>{{ numberValue(row.fn) }}</td>
                    <td>{{ numberValue(row.tn) }}</td>
                  </tr>
                </tbody>
              </table>
            </article>
            <div v-else class="empty-card">暂无结果</div>
          </template>
        </section>

        <section v-if="activeModule === 'cluster'" class="module-panel">
          <div class="tab-bar">
            <button type="button" :class="{ active: activeClusterTab === 'manage' }" @click="changeTab('cluster', 'manage')">聚类管理</button>
            <button type="button" :class="{ active: activeClusterTab === 'history' }" @click="changeTab('cluster', 'history')">分析列表</button>
          </div>

          <template v-if="activeClusterTab === 'manage'">
            <article class="toolbar-card">
              <label>
                <span>分组数</span>
                <input v-model.number="clusterCount" type="number" min="2" max="20" />
              </label>
              <button class="primary-btn" @click="runCluster">生成分组</button>
            </article>
            <template v-if="state.cluster?.result">
              <div class="card-grid four">
                <article class="metric-card">
                  <span>分组数</span>
                  <strong>{{ state.cluster.result.clusterCount }}</strong>
                </article>
                <article class="metric-card">
                  <span>图库图片</span>
                  <strong>{{ numberValue(state.cluster.result.totalImages) }}</strong>
                </article>
                <article class="metric-card">
                  <span>组内紧凑度</span>
                  <strong>{{ state.cluster.result.inertia }}</strong>
                </article>
                <article class="metric-card">
                  <span>分析记录</span>
                  <strong>{{ state.cluster.result.runCode || "--" }}</strong>
                </article>
              </div>
              <article class="detail-card cluster-report-summary">
                <span>分析结论</span>
                <strong>{{ state.cluster.result.report?.assessment }}</strong>
              </article>
              <div class="cluster-grid">
                <article v-for="group in state.cluster.result.groups.items" :key="group.clusterId" class="cluster-card">
                  <header>
                    <strong>分组 {{ group.clusterId }}</strong>
                    <span>{{ group.count }}</span>
                  </header>
                  <div class="cluster-images">
                    <div v-for="item in group.items" :key="item.id" class="cluster-image">
                      <img :src="item.thumbnail" :alt="item.name" />
                      <span>{{ item.name }}</span>
                    </div>
                  </div>
                </article>
              </div>
              <div class="pager">
                <button :disabled="state.cluster.result.groups.page <= 1" @click="changePage('cluster', state.cluster.result.groups.page - 1)">
                  上一页
                </button>
                <span>{{ state.cluster.result.groups.page }} / {{ state.cluster.result.groups.totalPages }}</span>
                <button
                  :disabled="state.cluster.result.groups.page >= state.cluster.result.groups.totalPages"
                  @click="changePage('cluster', state.cluster.result.groups.page + 1)"
                >
                  下一页
                </button>
              </div>
            </template>
            <div v-else class="empty-card">暂无结果</div>
          </template>

          <template v-else>
            <article class="table-card cluster-history-card">
              <div class="table-card-header">
                <strong>聚类分析列表</strong>
                <span>共{{ state.cluster?.history?.total ?? 0 }}条</span>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>分析记录</th>
                    <th>分组数</th>
                    <th>图库图片</th>
                    <th>组内紧凑度</th>
                    <th>单图紧凑度</th>
                    <th>最大组/最小组</th>
                    <th>分析结论</th>
                    <th>创建时间</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in state.cluster?.history?.items ?? []" :key="item.runId">
                    <td>
                      <button type="button" class="table-link-btn" @click="openClusterDetail(item.runId)">{{ item.runCode || "--" }}</button>
                    </td>
                    <td>{{ item.clusterCount }}</td>
                    <td>{{ item.totalImages }}</td>
                    <td>{{ item.inertia }}</td>
                    <td>{{ item.inertiaPerImage }}</td>
                    <td>{{ item.largestGroup }} / {{ item.smallestGroup }}</td>
                    <td>{{ item.assessment }}</td>
                    <td>{{ item.createdAt }}</td>
                    <td>
                      <button type="button" class="ghost-btn" @click="openClusterDetail(item.runId)">查看报告</button>
                    </td>
                  </tr>
                  <tr v-if="!(state.cluster?.history?.items ?? []).length">
                    <td colspan="9">暂无分析记录</td>
                  </tr>
                </tbody>
              </table>
            </article>
            <div class="pager">
              <button :disabled="state.cluster?.history?.page <= 1" @click="changePage('clusterHistory', 1)">首页</button>
              <button :disabled="state.cluster?.history?.page <= 1" @click="changePage('clusterHistory', state.cluster?.history?.page - 1)">
                上一页
              </button>
              <span>{{ state.cluster?.history?.page }} / {{ state.cluster?.history?.totalPages }}</span>
              <button
                :disabled="state.cluster?.history?.page >= state.cluster?.history?.totalPages"
                @click="changePage('clusterHistory', state.cluster?.history?.page + 1)"
              >
                下一页
              </button>
              <button
                :disabled="state.cluster?.history?.page >= state.cluster?.history?.totalPages"
                @click="changePage('clusterHistory', state.cluster?.history?.totalPages)"
              >
                末页
              </button>
              <div class="pager-jump">
                <input v-model="pageJumpInputs.clusterHistory" type="number" min="1" :max="state.cluster?.history?.totalPages" />
                <button type="button" @click="jumpPage('clusterHistory', state.cluster?.history?.totalPages)">跳转</button>
              </div>
            </div>
          </template>

          <div v-if="state.cluster?.detail" class="dialog-backdrop" @click.self="closeClusterDetail">
            <section class="dialog-panel cluster-detail-dialog">
              <header class="dialog-header">
                <strong>聚类分析报告 {{ state.cluster.detail.runCode || "--" }}</strong>
                <button class="ghost-btn" @click="closeClusterDetail">关闭</button>
              </header>
              <div class="dialog-body">
                <div class="card-grid four">
                  <article class="metric-card">
                    <span>分析图片</span>
                    <strong>{{ state.cluster.detail.totalImages }}</strong>
                  </article>
                  <article class="metric-card">
                    <span>分组数</span>
                    <strong>{{ state.cluster.detail.clusterCount }}</strong>
                  </article>
                  <article class="metric-card">
                    <span>平均组大小</span>
                    <strong>{{ state.cluster.detail.report?.averageGroupSize }}</strong>
                  </article>
                  <article class="metric-card">
                    <span>单图紧凑度</span>
                    <strong>{{ state.cluster.detail.report?.inertiaPerImage }}</strong>
                  </article>
                </div>
                <article class="detail-card cluster-report-summary">
                  <span>报告结论</span>
                  <strong>{{ state.cluster.detail.report?.assessment }}</strong>
                  <small>创建时间 {{ state.cluster.detail.createdAt }}，创建人 {{ state.cluster.detail.createdBy }}</small>
                </article>
                <div class="card-grid three cluster-report-grid">
                  <article class="detail-card">
                    <span>最大分组</span>
                    <strong>{{ state.cluster.detail.report?.largestGroup }}张</strong>
                  </article>
                  <article class="detail-card">
                    <span>最小分组</span>
                    <strong>{{ state.cluster.detail.report?.smallestGroup }}张</strong>
                  </article>
                  <article class="detail-card">
                    <span>类别覆盖</span>
                    <strong>{{ state.cluster.detail.report?.labelCount }}类</strong>
                  </article>
                </div>
                <article class="detail-card">
                  <span>主要类别</span>
                  <div class="result-tag-row">
                    <span v-for="item in state.cluster.detail.report?.topLabels ?? []" :key="item.label">
                      {{ item.label }} {{ item.count }}张
                    </span>
                  </div>
                </article>
                <article class="table-card cluster-report-table">
                  <div class="table-card-header">
                    <strong>分组摘要</strong>
                  </div>
                  <table>
                    <thead>
                      <tr>
                        <th>分组</th>
                        <th>图片数</th>
                        <th>主要类别</th>
                        <th>主要类别占比</th>
                        <th>类别分布</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="item in state.cluster.detail.report?.groupSummaries ?? []" :key="item.clusterId">
                        <td>分组 {{ item.clusterId }}</td>
                        <td>{{ item.count }}</td>
                        <td>{{ item.dominantLabel }}</td>
                        <td>{{ percentValue(item.dominantRatio) }}</td>
                        <td>
                          <div class="label-distribution-cell">
                            <span v-for="label in item.labelDistribution" :key="`${item.clusterId}-${label.label}`">
                              {{ label.label }} {{ label.count }}
                            </span>
                          </div>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </article>
              </div>
            </section>
          </div>
        </section>

        <section v-if="activeModule === 'metrics'" class="module-panel">
          <div class="card-grid four">
            <article class="metric-card">
              <span>图库总量</span>
              <strong>{{ numberValue(state.metrics?.overview?.galleryCount) }}</strong>
            </article>
            <article class="metric-card">
              <span>检索次数</span>
              <strong>{{ numberValue(state.metrics?.overview?.queryCount) }}</strong>
            </article>
            <article class="metric-card">
              <span>平均 mAP@K</span>
              <strong>{{ numberValue(state.metrics?.overview?.averageMapAtK) }}</strong>
            </article>
            <article class="metric-card">
              <span>平均 Recall@K</span>
              <strong>{{ numberValue(state.metrics?.overview?.averageRecallAtK) }}</strong>
            </article>
          </div>
          <div class="card-grid four">
            <article v-for="item in state.metrics?.methods ?? []" :key="item.method" class="metric-card">
              <span>{{ item.method }}</span>
              <strong>{{ numberValue(item.averageElapsedMs) }} ms</strong>
              <small>{{ item.indexSizeBytes }} bytes / {{ item.runs }} 次</small>
            </article>
          </div>
        </section>

        <section v-if="activeModule === 'profile'" class="module-panel">
          <article class="form-card profile-card">
            <label>
              <span>账号</span>
              <input :value="state.profile?.username" disabled />
            </label>
            <label>
              <span>姓名</span>
              <input v-model="profileForm.displayName" />
            </label>
            <label>
              <span>手机号</span>
              <input v-model="profileForm.phone" />
            </label>
            <label>
              <span>邮箱</span>
              <input v-model="profileForm.email" />
            </label>
            <label>
              <span>所属单位</span>
              <input v-model="profileForm.organization" />
            </label>
            <button class="primary-btn" @click="updateProfile">保存</button>
          </article>
        </section>
      </main>
    </section>
  </div>
</template>
