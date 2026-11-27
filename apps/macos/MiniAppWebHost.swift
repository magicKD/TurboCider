import AppKit
import SwiftUI
import UniformTypeIdentifiers
import WebKit

struct MiniAppWebView: NSViewRepresentable {
    let package: MiniAppPackage
    var runtime: MiniAppRuntime

    func makeCoordinator() -> MiniAppWebHost { MiniAppWebHost(package: package, runtime: runtime) }
    func makeNSView(context: Context) -> WKWebView { context.coordinator.makeView() }
    func updateNSView(_ webView: WKWebView, context: Context) {}
    static func dismantleNSView(_ webView: WKWebView, coordinator: MiniAppWebHost) { coordinator.disconnect(webView) }
}

@MainActor
final class MiniAppWebHost: NSObject, WKURLSchemeHandler, WKScriptMessageHandlerWithReply, WKNavigationDelegate, WKUIDelegate {
    private let package: MiniAppPackage
    private weak var runtime: MiniAppRuntime?
    private weak var activeView: WKWebView?
    private var materialTasks: [ObjectIdentifier: Task<Void, Never>] = [:]
    private static let channel = "turbocider"
    private static let policy = [
        "default-src 'none'", "script-src tcapp:", "style-src tcapp: 'unsafe-inline'",
        "img-src tcapp: blob: data:", "font-src tcapp:", "connect-src tcapp:",
        "media-src tcapp: blob:", "frame-src 'none'", "form-action 'none'",
        "worker-src 'none'", "object-src 'none'", "base-uri 'none'"
    ].joined(separator: "; ")
    private static let blockNetwork = """
    [
      {"trigger":{"url-filter":"^http://"},"action":{"type":"block"}},
      {"trigger":{"url-filter":"^https://"},"action":{"type":"block"}},
      {"trigger":{"url-filter":"^ws://"},"action":{"type":"block"}},
      {"trigger":{"url-filter":"^wss://"},"action":{"type":"block"}},
      {"trigger":{"url-filter":"^ftp://"},"action":{"type":"block"}}
    ]
    """
    private static let sdkBridge = """
    (() => {
      const post = window.webkit.messageHandlers.turbocider;
      const call = async (method, payload = {}) => {
        const response = await post.postMessage({ method, payload });
        if (!response.ok) {
          const error = new Error(response.message);
          error.code = response.code;
          throw error;
        }
        return response.value;
      };
      const base64 = blob => new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
      const createRequestID = () => {
        const bytes = crypto.getRandomValues(new Uint8Array(16));
        bytes[6] = (bytes[6] & 15) | 64;
        bytes[8] = (bytes[8] & 63) | 128;
        const h = Array.from(bytes, n => n.toString(16).padStart(2, '0')).join('');
        return [h.slice(0, 8), h.slice(8, 12), h.slice(12, 16), h.slice(16, 20), h.slice(20)].join('-');
      };
      Object.defineProperty(window, 'TurboCider', {
        value: Object.freeze({
          sdkVersion: 1, call, createRequestID,
          media: Object.freeze({
            bundled: path => call('media.bundled', { path }),
            get: id => call('media.get', { id }),
            import: async (blob, name) => {
              if (blob.size > 16 * 1024 * 1024) throw new Error('单张图片上限为 16 MiB。');
              return call('media.import', { data: await base64(blob), name: name || blob.name || 'image' });
            },
            save: id => call('media.export', { id }),
            saveBlob: async (blob, name) => {
              if (blob.size > 24 * 1024 * 1024) throw new Error('单次导出上限为 24 MiB。');
              return call('media.export', { data: await base64(blob), name });
            }
          }),
          storage: Object.freeze({
            get: key => call('storage.get', { key }),
            set: (key, value) => call('storage.set', { key, value })
          }),
          workflow: Object.freeze({ plan: options => call('workflow.plan', options) }),
          jobs: Object.freeze({
            submit: id => call('jobs.submit', { id }),
            get: id => call('jobs.get', { id }),
            async *watch(id, { signal } = {}) {
              const terminal = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
              for (;;) {
                signal?.throwIfAborted();
                const state = await call('jobs.get', { id });
                yield state;
                if (terminal.has(state.state)) return;
                await new Promise(resolve => setTimeout(resolve, 550));
              }
            },
            list: () => call('jobs.list'),
            cancel: id => call('jobs.cancel', { id }),
            remove: id => call('jobs.remove', { id })
          })
        })
      });
    })();
    """

    init(package: MiniAppPackage, runtime: MiniAppRuntime) {
        self.package = package
        self.runtime = runtime
    }

    func makeView() -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = WKWebsiteDataStore.nonPersistent()
        configuration.setURLSchemeHandler(self, forURLScheme: "tcapp")
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        configuration.userContentController.addScriptMessageHandler(self, contentWorld: .page, name: Self.channel)
        let saved = (try? JSONEncoder().encode(runtime?.storageSnapshot(package) ?? [:]))?.base64EncodedString() ?? "e30="
        let storageBootstrap = """
        Object.defineProperty(window, 'TurboCiderStorage', {
          value: JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('\(saved)'), c => c.charCodeAt(0))))
        });
        """
        configuration.userContentController.addUserScript(WKUserScript(source: Self.sdkBridge + "\n" + storageBootstrap,
            injectionTime: .atDocumentStart, forMainFrameOnly: true))
        let view = WKWebView(frame: .zero, configuration: configuration)
        activeView = view
        view.navigationDelegate = self
        view.uiDelegate = self
        view.allowsBackForwardNavigationGestures = false
        if package.builtIn, #available(macOS 13.3, *) {
            view.isInspectable = ProcessInfo.processInfo.environment["TURBOCIDER_MINI_APP_INSPECT"] == "1"
        }
        WKContentRuleListStore.default().compileContentRuleList(forIdentifier: "TurboCiderMiniAppNetworkV1",
            encodedContentRuleList: Self.blockNetwork) { [weak self, weak view] rules, error in
                DispatchQueue.main.async {
                    guard let self, let view, self.activeView === view else { return }
                    if let rules {
                        view.configuration.userContentController.add(rules)
                        if let entry = URL(string: "tcapp://\(self.package.id)/\(self.package.manifest.entry)") {
                            view.load(URLRequest(url: entry))
                        }
                    } else {
                        self.runtime?.message = "应用网络规则加载失败。\(error?.localizedDescription ?? "")"
                    }
                }
            }
        return view
    }

    func disconnect(_ webView: WKWebView) {
        materialTasks.values.forEach { $0.cancel() }
        materialTasks.removeAll()
        webView.configuration.userContentController.removeScriptMessageHandler(forName: Self.channel)
        webView.stopLoading()
        webView.uiDelegate = nil
        webView.navigationDelegate = nil
        activeView = nil
    }

    private func owns(_ url: URL?) -> Bool {
        guard let url else { return false }
        return url.scheme == "tcapp" && url.host == package.id && url.user == nil && url.password == nil && url.port == nil
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard action.targetFrame?.isMainFrame == true, owns(action.request.url),
              action.request.url?.pathExtension == "html" else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        guard frame.isMainFrame, owns(frame.request.url), package.manifest.permissions.contains(.importMedia) else {
            completionHandler(nil)
            return
        }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]; panel.canChooseDirectories = false
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.title = "选择用于\(package.manifest.name)的图片"
        if let window = webView.window {
            panel.beginSheetModal(for: window) { completionHandler($0 == .OK ? panel.urls : nil) }
        } else {
            panel.begin { completionHandler($0 == .OK ? panel.urls : nil) }
        }
    }

    func webView(_ webView: WKWebView, start task: any WKURLSchemeTask) {
        guard let url = task.request.url, owns(url), ["GET", "HEAD"].contains(task.request.httpMethod ?? "GET") else {
            task.didFailWithError(URLError(.unsupportedURL)); return
        }
        let path = String(url.path.dropFirst())
        if path.hasPrefix("_materials/"), package.builtIn, let runtime {
            let key = ObjectIdentifier(task as AnyObject)
            let token = String(path.dropFirst("_materials/".count))
            let package = self.package
            materialTasks[key] = Task { [weak self] in
                let resource = try? await runtime.materialResource(token, app: package)
                guard let self, self.materialTasks.removeValue(forKey: key) != nil, !Task.isCancelled else { return }
                self.serve(task, url: url, body: resource?.data, mime: resource?.mime)
            }
            return
        }
        var body: Data?
        var mime: String?
        if path.hasPrefix("_media/"), let id = UUID(uuidString: String(path.dropFirst(7))),
           let (file, type) = try? runtime?.mediaFile(id, package: package) {
            body = try? Data(contentsOf: file, options: .mappedIfSafe)
            mime = type
        } else if MiniAppPackage.validPath(path) {
            body = package.files[path]
            mime = MiniAppPackage.mimeTypes[(path as NSString).pathExtension.lowercased()]
        }
        serve(task, url: url, body: body, mime: mime)
    }

    private func serve(_ task: any WKURLSchemeTask, url: URL, body: Data?, mime: String?) {
        let contentType = mime ?? "text/plain"
        let headers = ["Content-Type": contentType, "Content-Security-Policy": Self.policy,
            "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*"]
        guard let response = HTTPURLResponse(url: url, statusCode: body != nil ? 200 : 404, httpVersion: "HTTP/1.1", headerFields: headers) else {
            task.didFailWithError(URLError(.badURL)); return
        }
        task.didReceive(response)
        if let body, task.request.httpMethod != "HEAD" { task.didReceive(body) }
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: any WKURLSchemeTask) {
        materialTasks.removeValue(forKey: ObjectIdentifier(task as AnyObject))?.cancel()
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage,
                               replyHandler: @escaping (Any?, String?) -> Void) {
        guard message.name == Self.channel, message.frameInfo.isMainFrame,
              message.webView === activeView, owns(message.frameInfo.request.url),
              let runtime, let envelope = message.body as? [String: Any],
              let method = envelope["method"] as? String, method.utf8.count <= 80,
              let payload = envelope["payload"] as? [String: Any],
              JSONSerialization.isValidJSONObject(payload) else {
            replyHandler(["ok": false, "code": "invalidOrigin", "message": "应用调用来源或格式无效。"], nil)
            return
        }
        let isMedia = method == "media.import" || method == "media.export"
        let maxRequest = isMedia ? 34 * 1024 * 1024 : 256 * 1024
        guard let bytes = try? JSONSerialization.data(withJSONObject: payload, options: .withoutEscapingSlashes),
              bytes.count <= maxRequest else {
            replyHandler(["ok": false, "code": "requestTooLarge", "message": "应用请求超出大小上限。"], nil)
            return
        }
        Task {
            do { replyHandler(["ok": true, "value": try await runtime.call(method, payload: payload, app: package)], nil) }
            catch let error as MiniAppRuntime.Failure {
                replyHandler(["ok": false, "code": error.code, "message": error.message], nil)
            } catch {
                replyHandler(["ok": false, "code": "runtimeError", "message": MiniAppRuntime.redact(error.localizedDescription)], nil)
            }
        }
    }
}

struct ApplicationsView<NativeTools: View>: View {
    @ObservedObject var runtime: MiniAppRuntime
    @State private var selected: String?
    @State private var nativeToolsSelected = false
    let showModels: () -> Void
    let showTasks: () -> Void
    @ViewBuilder var nativeTools: () -> NativeTools

    var body: some View {
        VStack(spacing: 0) {
            if let selected, let package = runtime.package(selected) {
                HStack {
                    back { self.selected = nil }
                    Label(package.manifest.name, systemImage: package.builtIn ? "person.crop.rectangle" : "app")
                        .font(.headline)
                    Spacer()
                    if package.builtIn {
                        Button("选择 Turbo 模型…") {
                            Task {
                                do {
                                    if try await runtime.chooseTurboModel() { runtime.message = "已选择本机 Qwen Image 2.1 Turbo。" }
                                } catch { runtime.message = error.localizedDescription }
                            }
                        }
                    }
                    Button("任务", action: showTasks)
                }.padding(12)
                Divider()
                MiniAppWebView(package: package, runtime: runtime).id(package.digest)
            } else if nativeToolsSelected {
                HStack {
                    back { nativeToolsSelected = false }
                    Text("创作模板").font(.headline)
                    Spacer()
                }.padding(12)
                Divider()
                nativeTools()
            } else {
                catalog
            }
        }
        .alert("应用", isPresented: Binding(get: { runtime.message != nil }, set: { if !$0 { runtime.message = nil } })) {
            Button("好", role: .cancel) { runtime.message = nil }
        } message: { Text(runtime.message ?? "") }
    }

    private func back(_ action: @escaping () -> Void) -> some View {
        Button(action: action) { Label("所有应用", systemImage: "chevron.left") }
    }

    private var catalog: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    Text("应用").font(.largeTitle.weight(.semibold))
                    Spacer()
                    Button("模型", action: showModels)
                    Button("安装应用…", action: choosePackage)
                        .accessibilityIdentifier("installMiniApp")
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 280), spacing: 16)], alignment: .leading, spacing: 16) {
                    ForEach(runtime.packages) { package in
                        tile(title: package.manifest.name, detail: package.manifest.description ?? "版本 \(package.manifest.version)",
                             icon: package.builtIn ? "person.crop.rectangle" : "app", subtitle: package.builtIn ? "官方" : "社区") {
                            selected = package.id
                        }.contextMenu {
                            if !package.builtIn { Button("卸载应用…", role: .destructive) { uninstall(package) } }
                        }.accessibilityIdentifier("openMiniApp-\(package.id)")
                    }
                    tile(title: "创作模板", detail: "换装、人物一致性、换脸、扩图和透明图。", icon: "slider.horizontal.3", subtitle: "内置工具") {
                        nativeToolsSelected = true
                    }
                }
            }.padding(26)
        }
    }

    private func tile(title: String, detail: String, icon: String, subtitle: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                Label(title, systemImage: icon).font(.title3.weight(.semibold))
                Text(detail).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }.padding(16).frame(maxWidth: .infinity, minHeight: 122, alignment: .leading)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
        }.buttonStyle(.plain)
    }

    private func choosePackage() {
        let panel = NSOpenPanel()
        panel.title = "安装社区应用"
        panel.allowedContentTypes = [UTType(filenameExtension: "tcapp") ?? .zip]
        panel.allowsMultipleSelection = false; panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do {
                let package = try await Task.detached { try MiniAppPackage(source: url) }.value
                let alert = NSAlert()
                let update = runtime.package(package.id) != nil
                alert.messageText = update ? "更新 \(package.manifest.name)？" : "安装 \(package.manifest.name)？"
                let permissions = package.manifest.permissions.map { "• " + $0.title }.joined(separator: "\n")
                alert.informativeText = "\(package.manifest.id) · \(package.manifest.version)" +
                    (permissions.isEmpty ? "" : "\n\n\(permissions)") +
                    (update ? "\n\n更新后应用可继续访问其现有数据。" : "")
                alert.addButton(withTitle: update ? "更新" : "安装")
                alert.addButton(withTitle: "取消")
                guard alert.runModal() == .alertFirstButtonReturn else { return }
                try runtime.install(package)
                selected = package.id
            } catch { runtime.message = error.localizedDescription }
        }
    }

    private func uninstall(_ package: MiniAppPackage) {
        let alert = NSAlert()
        alert.messageText = "卸载 \(package.manifest.name)？"
        alert.informativeText = "已提交的任务和图片仍可在任务记录中查看。"
        alert.addButton(withTitle: "取消")
        alert.addButton(withTitle: "卸载")
        guard alert.runModal() == .alertSecondButtonReturn else { return }
        do { try runtime.uninstall(package) }
        catch { runtime.message = error.localizedDescription }
    }
}
