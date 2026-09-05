import Foundation
import TurboCiderKit

@main
struct TurboCiderSwiftCLI {
    static func main() async {
        let client = TurboCiderClient()
        let daemon = TurboCiderDaemon()
        let selfStart = CommandLine.arguments.contains("--self-start")
        do {
            if selfStart {
                try await daemon.ensureRunning(client: client)
            }
            let models = try await client.models()
            for model in models {
                print("\(model.id)\t\(model.engine)\t\(model.name)")
            }
            if selfStart {
                let system = try await client.system()
                print("daemon_pid\t\(daemon.ownedProcessID ?? 0)")
                print("available_models\t\(system.models.filter(\.available).count)")
                daemon.stop()
            }
        } catch {
            daemon.stop()
            fputs("turbocider-swift: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
