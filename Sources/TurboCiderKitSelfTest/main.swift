import Foundation
import TurboCiderKit

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        fputs("FAIL: \(message)\n", stderr)
        exit(1)
    }
}

do {
    let request = TCGenerationRequest(
        model: "minimax-h3-turbo",
        task: "video",
        prompt: "fox",
        mode: "text_to_video",
        output: TCOutputSpec(type: "video"),
        policy: TCPolicySpec(execution: .gpuANE, allowFallback: false),
        engineOptions: [
            "h3": [
                "super": true,
                "args": ["--super-profile", "480p"],
                "env": ["H3_PRIVATE_ANE_QKV_CHECKPOINT": "1"],
            ]
        ]
    )
    let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
    let policy = object["policy"] as! [String: Any]
    require(policy["execution"] as? String == "gpu_ane", "execution wire value")
    require(policy["allow_fallback"] as? Bool == false, "fallback wire key")
    require(object["mode"] as? String == "text_to_video", "generation mode wire value")
    let engineOptions = object["engine_options"] as! [String: Any]
    let h3 = engineOptions["h3"] as! [String: Any]
    require(h3["super"] as? Bool == true, "engine options boolean")
    require((h3["args"] as? [String]) == ["--super-profile", "480p"], "engine options array")
    let environment = h3["env"] as! [String: Any]
    require(
        environment["H3_PRIVATE_ANE_QKV_CHECKPOINT"] as? String == "1",
        "engine options nested object"
    )

    let recursiveOptions = try JSONDecoder().decode(
        [String: TCJSONValue].self,
        from: Data("""
        {"flux2":{"dynamic_text_length":false,"ane_blocks":[0,1],"runtime":{"mode":"warm","limit":null}}}
        """.utf8)
    )
    require(
        recursiveOptions["flux2"] == [
            "dynamic_text_length": false,
            "ane_blocks": [0, 1],
            "runtime": ["mode": "warm", "limit": nil],
        ],
        "recursive JSON value decoding"
    )

    let modelsData = Data("""
    {"data":[{"id":"flux2-klein-4b","name":"FLUX.2 Klein 4B","engine":"flux2","version":"0.1","capabilities":{"tasks":["image"],"inputs":["text","image"],"reference_roles":["init_image","reference"],"modes":["text_to_image","image_to_image","image_edit"],"max_reference_images":8,"profiles":["quality"],"execution":["gpu","gpu_ane"],"audio_output":false,"audio_required":false,"recommended_width":512,"recommended_height":512,"recommended_frames":1,"recommended_fps":24,"recommended_steps":4,"recommended_persistent":true},"plans":[{"id":"flux2.mlx","execution":"gpu","quality":"exact","profile":"*","production":true,"description":"Portable MLX backend"}]}]}
    """.utf8)
    let models = try JSONDecoder().decode(TCModelsResponse.self, from: modelsData)
    require(models.data.count == 1, "model descriptor decoding")
    require(models.data[0].capabilities.recommendedWidth == 512, "recommended width decoding")
    require(models.data[0].capabilities.recommendedFrames == 1, "recommended frames decoding")
    require(models.data[0].capabilities.audioOutput == false, "audio capability decoding")
    require(models.data[0].capabilities.audioRequired == false, "audio requirement decoding")
    require(models.data[0].capabilities.referenceRoles == ["init_image", "reference"], "reference roles decoding")
    require(models.data[0].capabilities.modes == ["text_to_image", "image_to_image", "image_edit"], "generation modes decoding")
    require(models.data[0].capabilities.maxReferenceImages == 8, "reference image limit decoding")
    require(models.data[0].capabilities.recommendedPersistent == true, "persistent recommendation decoding")
    require(models.data[0].plans?.first?.execution == .gpu, "model plan decoding")

    let plansData = Data("""
    {"data":[{"plan":{"id":"flux2.mlx_ane.persistent","execution":"gpu_ane","quality":"validated","profile":"*","production":true,"priority":130,"description":"Persistent hybrid","requirements":{"persistent":true,"shapes":[{"width":512,"height":512}]},"environment":{},"metadata":{"expected_seconds_warm":2.3}},"failures":[],"available":true}]}
    """.utf8)
    let plans = try JSONDecoder().decode(TCPlansResponse.self, from: plansData)
    require(plans.data.first?.available == true, "plan availability decoding")
    require(plans.data.first?.plan.execution == .gpuANE, "plan execution decoding")
    require(
        plans.data.first?.plan.requirements["persistent"] == true,
        "plan requirements decoding"
    )
    require(
        plans.data.first?.plan.metadata["expected_seconds_warm"] == 2.3,
        "plan metadata decoding"
    )

    let data = Data("""
    {"id":"1","state":"succeeded","plan_id":"gpu","progress":1,"phase":"complete","elapsed_seconds":2.5,"estimated_total_seconds":2.5,"estimated_remaining_seconds":null,"output_paths":[],"error":null}
    """.utf8)
    let job = try JSONDecoder().decode(TCJobRecord.self, from: data)
    require(job.isTerminal, "terminal job state")
    require(job.elapsedSeconds == 2.5, "elapsed time decoding")
    require(TurboCiderDaemon.locatePackageRoot() != nil, "daemon package-root discovery")
    print("TurboCiderKit self-test: PASS")
} catch {
    fputs("FAIL: \(error)\n", stderr)
    exit(1)
}
