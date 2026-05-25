import { ref } from "vue"
import { Streamlit, type RenderData } from "streamlit-component-lib"

const DEFAULT_STATE = {
  appTitle: "相似图像检索系统",
  authenticated: false,
  authMode: "login",
  notice: null,
}

const isEmbedded = window.self !== window.top || new URLSearchParams(window.location.search).has("streamlitUrl")
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL ?? (isEmbedded ? "http://127.0.0.1:8000" : "")
const CLIENT_ROLE = (import.meta as any).env?.VITE_CLIENT_ROLE ?? ""
const CLIENT_PORT = (import.meta as any).env?.VITE_CLIENT_PORT ?? window.location.port
const TOKEN_KEY = "img-search-api-token"

export const backendState = ref<Record<string, any>>(DEFAULT_STATE)
let reconnectTimer: number | undefined
let pendingStandaloneAction: { type: string; payload: Record<string, any> } | null = null

const reportFrameHeight = () => {
  if (!isEmbedded) {
    return
  }
  window.requestAnimationFrame(() => {
    const rootHeight = document.documentElement.scrollHeight
    const bodyHeight = document.body?.scrollHeight ?? 0
    const viewportHeight = window.innerHeight || 0
    Streamlit.setFrameHeight(Math.max(rootHeight, bodyHeight, viewportHeight, 720))
  })
}

const onRender = (event: Event) => {
  const detail = (event as CustomEvent<RenderData>).detail
  backendState.value = detail.args.state ?? {}
  reportFrameHeight()
}

const getToken = () => window.localStorage.getItem(TOKEN_KEY)
const setToken = (token: string | null | undefined) => {
  if (!token) {
    window.localStorage.removeItem(TOKEN_KEY)
    return
  }
  window.localStorage.setItem(TOKEN_KEY, token)
}

const applyStandaloneResponse = (payload: { token?: string | null; state?: Record<string, any> }) => {
  setToken(payload.token)
  backendState.value = payload.state ?? DEFAULT_STATE
}

export const apiUrl = (path: string) => `${API_BASE_URL}${path}`
const withClientContext = (payload: Record<string, any>) => ({
  ...payload,
  __clientRole: CLIENT_ROLE,
  __clientPort: CLIENT_PORT,
})

export const fetchApiJson = async (path: string) => {
  const token = getToken()
  const response = await fetch(apiUrl(path), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!response.ok) {
    throw new Error(`api request failed: ${response.status}`)
  }
  return response.json()
}

const scheduleStandaloneReconnect = () => {
  if (reconnectTimer || isEmbedded) return
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = undefined
    void fetchStandaloneState()
  }, 1200)
}

const fetchStandaloneState = async () => {
  try {
    const token = getToken()
    const response = await fetch(apiUrl("/api/state"), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!response.ok) {
      throw new Error(`state request failed: ${response.status}`)
    }
    const payload = await response.json()
    applyStandaloneResponse(payload)
    if (pendingStandaloneAction) {
      const action = pendingStandaloneAction
      pendingStandaloneAction = null
      sendEvent(action.type, action.payload)
    }
  } catch (error) {
    backendState.value = {
      ...DEFAULT_STATE,
      notice: {
        type: "error",
        message: "后端服务未连接",
      },
    }
    scheduleStandaloneReconnect()
  }
}

if (isEmbedded) {
  Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender as EventListener)
  window.addEventListener("resize", reportFrameHeight)
  window.addEventListener("load", reportFrameHeight, { once: true })
  Streamlit.setComponentReady()
  reportFrameHeight()
} else {
  void fetchStandaloneState()
}

export const sendEvent = async (type: string, payload: Record<string, any> = {}) => {
  if (!isEmbedded) {
    try {
      const token = getToken()
      const response = await fetch(apiUrl("/api/action"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ type, payload: withClientContext(payload) }),
      })
      if (!response.ok) {
        throw new Error(`action request failed: ${response.status}`)
      }
      const data = await response.json()
      applyStandaloneResponse(data)
      return data
    } catch (error) {
      pendingStandaloneAction = { type, payload }
      backendState.value = {
        ...backendState.value,
        notice: {
          type: "error",
          message: "后端服务未连接",
        },
      }
      scheduleStandaloneReconnect()
      throw error
    }
  }

  Streamlit.setComponentValue({
    eventId: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type,
    payload: withClientContext(payload),
  })
  return null
}
