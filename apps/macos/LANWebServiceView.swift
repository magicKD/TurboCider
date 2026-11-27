import AppKit
import CoreImage.CIFilterBuiltins
import SwiftUI

struct LANWebServiceView: View {
    @ObservedObject var web: LANWebServer
    let exclusiveServiceActive: Bool
    @State private var mode = LANWebServer.Mode.lan
    @State private var networkInterface = ""
    @State private var port = 8766

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Label("网页服务", systemImage: "qrcode").font(.title3.weight(.semibold))
                Text("在同一局域网中，手机、平板或其他电脑可以打开 TurboCider 工作台。创作与超分都在这台 Mac 上执行。")
                    .foregroundStyle(.secondary)
                HStack {
                    Picker("访问范围", selection: $mode) {
                        Text("仅本机").tag(LANWebServer.Mode.local)
                        Text("局域网").tag(LANWebServer.Mode.lan)
                    }.fixedSize()
                    if mode == .lan {
                        Picker("网卡", selection: $networkInterface) {
                            Text("自动").tag("")
                            ForEach(web.addresses) { address in Text("\(address.interface) · \(address.ip)").tag(address.interface) }
                        }.fixedSize()
                        Button("刷新网卡") { web.refreshAddresses() }
                    }
                    Text("端口")
                    TextField("8766", value: $port, format: .number.grouping(.never)).frame(width: 62)
                        .accessibilityIdentifier("webStudioPort")
                }.disabled(web.changing || web.mode != .off)
                Toggle("允许免配对直连", isOn: Binding(get: { web.allowsDirectAccess },
                                                   set: { web.setDirectAccess($0) }))
                    .disabled(web.changing)
                    .accessibilityIdentifier("allowDirectWebConnections")
                if web.allowsDirectAccess {
                    Text("同一网络中的设备可以直接使用创作和任务队列。关闭后，直连会话会失效。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Button(web.mode == .off ? "启动网页服务" : "关闭网页服务") {
                        if web.mode != .off { web.stop() }
                        else { Task { await web.start(mode, interface: networkInterface.isEmpty ? nil : networkInterface, port: port) } }
                    }.disabled(web.changing || (exclusiveServiceActive && web.mode == .off))
                        .accessibilityIdentifier("toggleWebStudio")
                    if web.changing { ProgressView().controlSize(.small) }
                    if web.mode == .off && mode == .lan {
                        Text("仅在可信局域网使用。HTTP 传输未加密。").font(.caption).foregroundStyle(.secondary)
                    }
                }
                if let url = web.url {
                    Divider()
                    HStack(alignment: .top, spacing: 16) {
                        if let link = web.allowsDirectAccess ? Self.appsURL(url) : web.pairingURL, let qr = Self.qrCode(link) {
                            Image(decorative: qr, scale: 1)
                                .interpolation(.none).resizable().scaledToFit().frame(width: 156, height: 156)
                                .padding(12).background(.white, in: RoundedRectangle(cornerRadius: 8))
                                .accessibilityLabel(web.allowsDirectAccess ? "TurboCider 网页直连二维码" : "TurboCider 网页配对二维码")
                        }
                        VStack(alignment: .leading, spacing: 8) {
                            LabeledContent("网页地址", value: url.absoluteString).font(.callout.monospaced()).textSelection(.enabled)
                            Text(web.allowsDirectAccess
                                 ? "扫码或打开网页地址即可进入工作台。"
                                 : "新设备请扫码或打开配对链接，并在此确认验证码。已配对的设备可直接打开网页地址。")
                                .font(.callout).foregroundStyle(.secondary)
                            if let expiry = web.pairingExpires {
                                Text("当前配对链接有效至 \(expiry.formatted(date: .omitted, time: .standard))")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            HStack {
                                if let link = web.pairingURL { Button("复制配对链接") { copy(link) } }
                                Button("复制网页地址") { copy(url) }
                                if !web.allowsDirectAccess { Button("刷新配对码") { web.issuePairingLink() } }
                            }
                        }
                    }
                }
                ForEach(web.approvals) { pending in
                    HStack {
                        Text(pending.device.name).lineLimit(1)
                        Text(pending.code).font(.body.monospacedDigit().weight(.semibold))
                        Spacer()
                        Button("拒绝") { web.reject(pending.id) }
                        Button("允许") { web.approve(pending.id) }.buttonStyle(.borderedProminent)
                            .accessibilityIdentifier("approveWebDevice")
                        Button("允许并授权管理") { web.approve(pending.id, canManage: true) }
                    }
                }
                if !web.devices.isEmpty {
                    DisclosureGroup("已配对设备（\(web.devices.count)）") {
                        ForEach(web.devices) { device in
                            HStack {
                                Text(device.name).lineLimit(1)
                                Spacer()
                                Toggle("可管理这台 Mac", isOn: Binding(
                                    get: { web.devices.first { $0.id == device.id }?.canManage == true },
                                    set: { web.setManagement($0, for: device.id) })).fixedSize()
                                Button("撤销配对") { web.revoke(device.id) }.buttonStyle(.borderless)
                            }
                        }
                    }
                }
                if let error = web.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
            }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear { web.refreshAddresses() }
        .accessibilityIdentifier("webStudioSection")
    }

    private static func appsURL(_ origin: URL) -> URL? {
        var url = URLComponents(url: origin, resolvingAgainstBaseURL: false)
        url?.fragment = "apps"
        return url?.url
    }

    static func qrCode(_ url: URL) -> CGImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(url.absoluteString.utf8)
        filter.correctionLevel = "M"
        guard let image = filter.outputImage?.transformed(by: CGAffineTransform(scaleX: 5, y: 5)) else { return nil }
        return CIContext().createCGImage(image, from: image.extent)
    }

    private func copy(_ url: URL) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(url.absoluteString, forType: .string)
    }
}
