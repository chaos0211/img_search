const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1"])
const SAFE_PROTOCOLS = new Set(["about:", "blob:", "data:"])
const guardMarker = "__imgSearchNetworkGuardInstalled__"
const isEmbedded = window.self !== window.top

type FetchInput = string | URL | Request | { url?: string } | null | undefined

const getBaseHref = (target: Window) => {
  try {
    return target.location.href
  } catch {
    return window.location.href
  }
}

const getOrigin = (target: Window) => {
  try {
    return target.location.origin
  } catch {
    return window.location.origin
  }
}

const normalizeUrl = (input: string | URL, baseHref: string) => {
  try {
    return new URL(String(input), baseHref)
  } catch {
    return null
  }
}

const resolveUrl = (input: FetchInput, baseHref: string) => {
  if (!input) {
    return null
  }
  if (typeof input === "object" && "url" in input && typeof input.url === "string") {
    return normalizeUrl(input.url, baseHref)
  }
  return normalizeUrl(input as string | URL, baseHref)
}

const isLoopback = (url: URL) => LOOPBACK_HOSTS.has(url.hostname)

const isAllowedUrl = (url: URL | null, allowedOrigin: string) => {
  if (!url) {
    return false
  }
  if (SAFE_PROTOCOLS.has(url.protocol)) {
    return true
  }
  if (url.origin === allowedOrigin) {
    return true
  }
  if ((url.protocol === "http:" || url.protocol === "https:" || url.protocol === "ws:" || url.protocol === "wss:") && isLoopback(url)) {
    return true
  }
  return false
}

const assertAllowed = (target: Window, kind: string, input: FetchInput | string | URL) => {
  const baseHref = getBaseHref(target)
  const allowedOrigin = getOrigin(target)
  const url = resolveUrl(input as FetchInput, baseHref)
  if (isAllowedUrl(url, allowedOrigin)) {
    return
  }
  const value = typeof input === "string" ? input : String(input)
  throw new Error(`[network-guard] blocked external ${kind}: ${value}`)
}

const patchImageSrc = (target: Window) => {
  const imageProto = target.HTMLImageElement?.prototype
  if (!imageProto) {
    return
  }
  const descriptor = Object.getOwnPropertyDescriptor(imageProto, "src")
  if (!descriptor?.set || !descriptor.get) {
    return
  }
  Object.defineProperty(imageProto, "src", {
    configurable: true,
    enumerable: descriptor.enumerable ?? true,
    get() {
      return descriptor.get!.call(this)
    },
    set(value: string) {
      assertAllowed(target, "image", value)
      descriptor.set!.call(this, value)
    },
  })
}

const installNetworkGuardOn = (target: Window) => {
  const globalScope = target as Window & Record<string, unknown>
  if (globalScope[guardMarker]) {
    return
  }
  globalScope[guardMarker] = true

  if (typeof target.fetch === "function") {
    const originalFetch = target.fetch.bind(target)
    target.fetch = ((input: FetchInput, init?: RequestInit) => {
      assertAllowed(target, "fetch", input)
      return originalFetch(input, init)
    }) as typeof target.fetch
  }

  if (typeof target.navigator?.sendBeacon === "function") {
    const originalSendBeacon = target.navigator.sendBeacon.bind(target.navigator)
    target.navigator.sendBeacon = ((url: string | URL, data?: BodyInit | null) => {
      if (!isAllowedUrl(resolveUrl(url, getBaseHref(target)), getOrigin(target))) {
        return false
      }
      return originalSendBeacon(url, data)
    }) as typeof target.navigator.sendBeacon
  }

  if (target.XMLHttpRequest?.prototype?.open) {
    const originalXhrOpen = target.XMLHttpRequest.prototype.open
    target.XMLHttpRequest.prototype.open = function open(
      method: string,
      url: string | URL,
      async?: boolean,
      username?: string | null,
      password?: string | null,
    ) {
      assertAllowed(target, "xhr", url)
      return originalXhrOpen.call(this, method, url, async ?? true, username, password)
    }
  }

  if (typeof target.WebSocket !== "undefined") {
    const OriginalWebSocket = target.WebSocket
    class GuardedWebSocket extends OriginalWebSocket {
      constructor(url: string | URL, protocols?: string | string[]) {
        assertAllowed(target, "websocket", url)
        super(url, protocols)
      }
    }
    target.WebSocket = GuardedWebSocket as typeof target.WebSocket
  }

  if (typeof target.EventSource !== "undefined") {
    const OriginalEventSource = target.EventSource
    class GuardedEventSource extends OriginalEventSource {
      constructor(url: string | URL, eventSourceInitDict?: EventSourceInit) {
        assertAllowed(target, "eventsource", url)
        super(url, eventSourceInitDict)
      }
    }
    target.EventSource = GuardedEventSource as typeof target.EventSource
  }

  patchImageSrc(target)
}

const installNetworkGuard = () => {
  const targets = new Set<Window>([window])

  // Do not monkey-patch the Streamlit host window from inside the component iframe.
  if (!isEmbedded) {
    for (const candidate of [window.parent, window.top]) {
      if (!candidate) {
        continue
      }
      try {
        void candidate.location.href
        targets.add(candidate)
      } catch {
        // Ignore inaccessible ancestor windows.
      }
    }
  }

  targets.forEach((target) => {
    try {
      installNetworkGuardOn(target)
    } catch (error) {
      console.warn("[network-guard] install failed", error)
    }
  })
}

installNetworkGuard()
