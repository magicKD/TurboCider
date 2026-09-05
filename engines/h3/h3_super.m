#include "h3_super.h"

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <dispatch/dispatch.h>
#import <Vision/Vision.h>

#import <CommonCrypto/CommonDigest.h>
#import <mach/mach.h>

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char *h3_super_default_sigmas =
    "0.909375, 0.725, 0.421875, 0.0";
static const char *h3_super_skip_0p9_sigmas = "0.725, 0.421875, 0.0";
static const char *h3_super_tail_0p42_sigmas = "0.421875, 0.0";
static const char *h3_super_official_stage2_revision =
    "bf86adedf518142442575d1ce2e767b7d01c8c76";
static const char *h3_super_official_transformer =
    "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors";
static const char *h3_super_official_transformer_sha256 =
    "2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b";
static const char *h3_super_official_bf16_transformer =
    "ltx-2.5-22b-dev-transformer-bf16.safetensors";
static const char *h3_super_official_bf16_transformer_sha256 =
    "792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584";
static const unsigned long long h3_super_official_bf16_transformer_bytes =
    42018190584ULL;
static const char *h3_super_official_refiner_lora =
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors";
static const char *h3_super_official_refiner_lora_sha256 =
    "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5";
static const char *h3_super_official_merged_refiner =
    "ltx-2.5-22b-dev-refiner-lora-0.8-comfy-int8-convrot.safetensors";
static const char *h3_super_official_merged_refiner_sha256 =
    "619ad78dc7ae321bd004288933eb8571b07db03d9a71abea24bb61f101aa7185";
static const char *h3_super_official_hybrid_refiner =
    "ltx-2.5-22b-dev-refiner-lora-0.8-video-self-bf16-"
    "int8-convrot.safetensors";
static const char *h3_super_official_hybrid_refiner_sha256 =
    "1b6e038ae87cf96b33492d4fb7ff62354f86d1b1523bce8633a436ca356bcae8";
static const unsigned long long h3_super_official_hybrid_refiner_bytes =
    24722047456ULL;
static const char *h3_super_handoff_magic = "H3SUPER_HANDOFF_V1\n";
enum { H3_SUPER_HANDOFF_HEADER_BYTES = 4096 };

/* The face-only gate keeps its compatibility envelope.  The production
 * person gate is intentionally stricter: matched runs showed that Stage-1
 * contour drift around 0.62--0.67 can be amplified above 0.9 by LTX, while
 * the clean front-safe candidate was 0.39 before and 0.46 after refinement. */
static const double h3_super_face_maximum_landmark = 0.80;
static const double h3_super_face_maximum_mouth = 0.75;
static const double h3_super_face_maximum_contour = 0.90;
static const double h3_super_person_maximum_landmark = 0.65;
static const double h3_super_person_maximum_mouth = 0.60;
static const double h3_super_person_maximum_contour = 0.55;
static const double h3_super_person_closeup_maximum_contour = 0.70;
static const double h3_super_output_closeup_maximum_contour = 0.80;
static const double h3_super_person_maximum_feature_mean = 0.75;
static const double h3_super_face_maximum_feature_mean = 0.65;
static const double h3_super_person_minimum_capture_quality = 0.10;
static const double h3_super_person_closeup_minimum_capture_quality = 0.12;
static const double h3_super_person_closeup_minimum_detection_rate = 0.90;
static const double h3_super_strict_minimum_capture_quality = 0.14;
static const double h3_super_strict_minimum_detection_rate = 0.98;
static const double h3_super_strict_maximum_center_jump = 0.02;
static const double h3_super_strict_maximum_log_area_jump = 0.10;
static const int h3_super_face_quality_stride = 6;
static const int h3_super_person_quality_stride = 3;
static const int h3_super_strict_person_quality_stride = 1;

typedef struct {
    int low;
    int high;
    float high_weight;
} h3_super_resize_axis;

static int h3_super_uses_merged_refiner(const h3_super_params *params) {
    return params && params->transformer &&
        (!strcmp(params->transformer, h3_super_official_merged_refiner) ||
         !strcmp(params->transformer, h3_super_official_hybrid_refiner));
}

static int h3_super_uses_hybrid_refiner(const h3_super_params *params) {
    return params && params->transformer &&
        !strcmp(params->transformer, h3_super_official_hybrid_refiner);
}

static int h3_super_uses_runtime_refiner_base(const h3_super_params *params) {
    return params && params->transformer &&
        (!strcmp(params->transformer, h3_super_official_transformer) ||
         !strcmp(params->transformer, h3_super_official_bf16_transformer));
}

static int h3_super_uses_bf16_transformer(const h3_super_params *params) {
    return params && params->transformer &&
        !strcmp(params->transformer, h3_super_official_bf16_transformer);
}

const char *h3_super_profile_name(h3_super_profile profile) {
    switch (profile) {
        case H3_SUPER_PROFILE_480P: return "480p";
        case H3_SUPER_PROFILE_480P_FAST: return "480p-fast";
        case H3_SUPER_PROFILE_480P_QUALITY: return "480p-quality";
        case H3_SUPER_PROFILE_480P_MOTION: return "480p-motion";
        case H3_SUPER_PROFILE_V2: return "v2";
    }
    return "unknown";
}

const char *h3_super_decoder_name(h3_super_decoder decoder) {
    switch (decoder) {
        case H3_SUPER_DECODER_TAEHV: return "taehv";
        case H3_SUPER_DECODER_OFFICIAL: return "official";
    }
    return "unknown";
}

const char *h3_super_stage2_schedule_name(h3_super_stage2_schedule schedule) {
    switch (schedule) {
        case H3_SUPER_STAGE2_SCHEDULE_DEFAULT: return "default";
        case H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9: return "skip-0p9";
        case H3_SUPER_STAGE2_SCHEDULE_TAIL_0P42: return "tail-0p42";
    }
    return "unknown";
}

static const char *h3_super_stage2_sigmas(
        h3_super_stage2_schedule schedule) {
    switch (schedule) {
        case H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9:
            return h3_super_skip_0p9_sigmas;
        case H3_SUPER_STAGE2_SCHEDULE_TAIL_0P42:
            return h3_super_tail_0p42_sigmas;
        case H3_SUPER_STAGE2_SCHEDULE_DEFAULT:
            return h3_super_default_sigmas;
    }
    return h3_super_default_sigmas;
}

static int h3_super_stage2_updates(h3_super_stage2_schedule schedule) {
    switch (schedule) {
        case H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9: return 2;
        case H3_SUPER_STAGE2_SCHEDULE_TAIL_0P42: return 1;
        case H3_SUPER_STAGE2_SCHEDULE_DEFAULT: return 3;
    }
    return 3;
}

void h3_super_params_apply_prefetch_defaults(
        h3_super_params *params,
        int start_step_given,
        int start_block_given) {
    if (!params || !params->prefetch_conditioning) return;
    if (!start_step_given && !start_block_given) {
        if (params->stage2_schedule == H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9) {
            params->prefetch_start_step = 0;
            params->prefetch_start_block = 15;
            if (!params->final_evict_blocks)
                params->final_evict_blocks = 15;
        } else {
            params->prefetch_start_step = 2;
            params->prefetch_start_block = 0;
        }
    }
    if (params->prefetch_start_block && !params->final_evict_blocks)
        params->final_evict_blocks = 5;
}

void h3_super_params_set_profile(h3_super_params *params,
                                 h3_super_profile profile) {
    if (!params) return;
    params->profile = profile;
    params->stage1_frames = 124;
    params->output_frames = 121;
    params->output_fps = 24;
    if (profile == H3_SUPER_PROFILE_V2) {
        params->stage1_width = 896;
        params->stage1_height = 512;
        params->input_width = 672;
        params->input_height = 384;
        params->refined_width = 1344;
        params->refined_height = 768;
        params->output_width = 1344;
        params->output_height = 768;
        params->decoder = H3_SUPER_DECODER_OFFICIAL;
        return;
    }
    if (profile == H3_SUPER_PROFILE_480P_MOTION) {
        /* Preserve more of the H3 trajectory in LTX instead of enlarging only
         * Stage 1 and then reducing it to the balanced 448x256 VAE input. */
        params->stage1_width = 576;
        params->stage1_height = 320;
        params->input_width = 512;
        params->input_height = 288;
        params->refined_width = 1024;
        params->refined_height = 576;
        params->output_width = 864;
        params->output_height = 480;
        params->decoder = H3_SUPER_DECODER_TAEHV;
        return;
    }
    if (profile == H3_SUPER_PROFILE_480P_FAST) {
        params->stage1_width = 448;
        params->stage1_height = 256;
    } else if (profile == H3_SUPER_PROFILE_480P_QUALITY) {
        params->stage1_width = 576;
        params->stage1_height = 320;
    } else {
        params->stage1_width = 512;
        params->stage1_height = 288;
    }
    /* LTX's VAE is 32x spatially compressed. Exact learned x2 therefore
     * requires a 32-aligned input and a 64-aligned decoded canvas. Decode at
     * 896x512, then perform the small delivery resize to exact 864x480. */
    params->input_width = 448;
    params->input_height = 256;
    params->refined_width = 896;
    params->refined_height = 512;
    params->output_width = 864;
    params->output_height = 480;
    params->decoder = H3_SUPER_DECODER_TAEHV;
}

void h3_super_params_apply_person_quality_preset(h3_super_params *params) {
    if (!params) return;
    h3_super_params_set_profile(params, H3_SUPER_PROFILE_V2);
    params->stage1_steps = 4;
    params->stage2_schedule = H3_SUPER_STAGE2_SCHEDULE_DEFAULT;
    params->transformer = h3_super_official_transformer;
    params->refiner_lora = h3_super_official_refiner_lora;
    params->refiner_lora_strength = 0.8;
    params->decoder = H3_SUPER_DECODER_TAEHV;
    params->taehv_compute_dtype = "bfloat16";
    params->taehv_temporal_mode = "sequential";
    params->taehv_keep_on_device = 0;
    params->sol_attention = 1;
    params->fuse_denoise_decode = 0;
    params->direct_handoff = 1;
    params->prepared_handoff = 1;
    params->face_quality_gate = 0;
    params->person_quality_gate = 1;
    params->output_quality_gate = 1;
    params->strict_person_quality = 1;
    params->quality_attempts = 3;
    params->min_face_area = 0.006;
    params->prefetch_transformer = 0;
    params->prefetch_conditioning = 0;
    params->prefetch_input_models = 0;
    params->prefetch_start_step = 0;
    params->prefetch_start_block = 0;
    params->final_evict_blocks = 0;
    params->allow_swap = 1;
}

static double h3_super_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static int h3_super_fail(char *error, size_t error_size,
                         const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static int h3_super_face_pixel_buffer(const float *rgb, int width, int height,
                                      CVPixelBufferRef *output,
                                      char *error, size_t error_size) {
    if (output) *output = NULL;
    if (!rgb || width < 1 || height < 1 || !output) {
        return h3_super_fail(error, error_size,
                             "invalid Stage-1 face-quality frame");
    }
    NSDictionary *attributes = @{
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };
    CVPixelBufferRef pixels = NULL;
    CVReturn status = CVPixelBufferCreate(
        kCFAllocatorDefault, (size_t)width, (size_t)height,
        kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)attributes, &pixels);
    if (status != kCVReturnSuccess || !pixels) {
        return h3_super_fail(error, error_size,
                             "cannot allocate Stage-1 face-quality buffer: %d",
                             (int)status);
    }
    status = CVPixelBufferLockBaseAddress(pixels, 0);
    if (status != kCVReturnSuccess) {
        CVPixelBufferRelease(pixels);
        return h3_super_fail(error, error_size,
                             "cannot lock Stage-1 face-quality buffer: %d",
                             (int)status);
    }
    uint8_t *base = CVPixelBufferGetBaseAddress(pixels);
    size_t row_bytes = CVPixelBufferGetBytesPerRow(pixels);
    for (int y = 0; y < height; y++) {
        uint8_t *row = base + (size_t)y * row_bytes;
        const float *source = rgb + (size_t)y * (size_t)width * 3;
        for (int x = 0; x < width; x++) {
            for (int channel = 0; channel < 3; channel++) {
                if (!isfinite(source[(size_t)x * 3 + (size_t)channel])) {
                    CVPixelBufferUnlockBaseAddress(pixels, 0);
                    CVPixelBufferRelease(pixels);
                    return h3_super_fail(
                        error, error_size,
                        "Stage-1 face-quality frame contains NaN or Inf");
                }
            }
            float red = fminf(1.0f, fmaxf(0.0f, source[(size_t)x * 3]));
            float green = fminf(
                1.0f, fmaxf(0.0f, source[(size_t)x * 3 + 1]));
            float blue = fminf(
                1.0f, fmaxf(0.0f, source[(size_t)x * 3 + 2]));
            row[(size_t)x * 4] = (uint8_t)lrintf(blue * 255.0f);
            row[(size_t)x * 4 + 1] = (uint8_t)lrintf(green * 255.0f);
            row[(size_t)x * 4 + 2] = (uint8_t)lrintf(red * 255.0f);
            row[(size_t)x * 4 + 3] = 255;
        }
    }
    CVPixelBufferUnlockBaseAddress(pixels, 0);
    *output = pixels;
    return 1;
}

typedef struct {
    int detected;
    int duplicate;
    int joint_count;
    int core_joint_count;
    int complete_chains;
    double joint_coverage;
    double core_coverage;
    double limb_asymmetry;
} h3_super_body_frame_metrics;

static const float h3_super_body_point_confidence = 0.15f;

static VNRecognizedPoint *h3_super_body_point(
        NSDictionary<VNHumanBodyPoseObservationJointName,
                     VNRecognizedPoint *> *points,
        VNHumanBodyPoseObservationJointName name) {
    VNRecognizedPoint *point = points[name];
    return point && point.confidence >= h3_super_body_point_confidence ?
        point : nil;
}

static double h3_super_body_point_distance(VNRecognizedPoint *a,
                                           VNRecognizedPoint *b) {
    if (!a || !b) return NAN;
    return hypot(a.location.x - b.location.x, a.location.y - b.location.y);
}

static int h3_super_body_frame_quality(
        NSArray<VNHumanBodyPoseObservation *> *observations,
        h3_super_body_frame_metrics *metrics,
        char *error, size_t error_size) {
    if (!metrics) {
        return h3_super_fail(error, error_size,
                             "invalid Stage-1 body-quality metrics");
    }
    memset(metrics, 0, sizeof(*metrics));
    metrics->duplicate = observations.count > 1;
    if (!observations.count) return 1;

    NSArray<VNHumanBodyPoseObservationJointName> *joint_names = @[
        VNHumanBodyPoseObservationJointNameNeck,
        VNHumanBodyPoseObservationJointNameRoot,
        VNHumanBodyPoseObservationJointNameLeftShoulder,
        VNHumanBodyPoseObservationJointNameLeftElbow,
        VNHumanBodyPoseObservationJointNameLeftWrist,
        VNHumanBodyPoseObservationJointNameRightShoulder,
        VNHumanBodyPoseObservationJointNameRightElbow,
        VNHumanBodyPoseObservationJointNameRightWrist,
        VNHumanBodyPoseObservationJointNameLeftHip,
        VNHumanBodyPoseObservationJointNameLeftKnee,
        VNHumanBodyPoseObservationJointNameLeftAnkle,
        VNHumanBodyPoseObservationJointNameRightHip,
        VNHumanBodyPoseObservationJointNameRightKnee,
        VNHumanBodyPoseObservationJointNameRightAnkle
    ];
    NSArray<VNHumanBodyPoseObservationJointName> *core_names = @[
        VNHumanBodyPoseObservationJointNameNeck,
        VNHumanBodyPoseObservationJointNameRoot,
        VNHumanBodyPoseObservationJointNameLeftShoulder,
        VNHumanBodyPoseObservationJointNameRightShoulder,
        VNHumanBodyPoseObservationJointNameLeftHip,
        VNHumanBodyPoseObservationJointNameRightHip
    ];
    NSDictionary<VNHumanBodyPoseObservationJointName,
                 VNRecognizedPoint *> *best_points = nil;
    int best_joint_count = -1;
    for (VNHumanBodyPoseObservation *observation in observations) {
        NSError *points_error = nil;
        NSDictionary *points = [observation
            recognizedPointsForGroupKey:
                VNHumanBodyPoseObservationJointsGroupNameAll
            error:&points_error];
        if (!points) {
            return h3_super_fail(
                error, error_size,
                "Apple Vision body joints failed: %s",
                points_error.localizedDescription.UTF8String);
        }
        int count = 0;
        for (VNHumanBodyPoseObservationJointName name in joint_names) {
            if (h3_super_body_point(points, name)) count++;
        }
        if (!best_points || count > best_joint_count) {
            best_points = points;
            best_joint_count = count;
        }
    }
    if (!best_points) return 1;

    int core_count = 0;
    for (VNHumanBodyPoseObservationJointName name in core_names) {
        if (h3_super_body_point(best_points, name)) core_count++;
    }
    NSArray<NSArray<VNHumanBodyPoseObservationJointName> *> *chains = @[
        @[VNHumanBodyPoseObservationJointNameLeftShoulder,
          VNHumanBodyPoseObservationJointNameLeftElbow,
          VNHumanBodyPoseObservationJointNameLeftWrist],
        @[VNHumanBodyPoseObservationJointNameRightShoulder,
          VNHumanBodyPoseObservationJointNameRightElbow,
          VNHumanBodyPoseObservationJointNameRightWrist],
        @[VNHumanBodyPoseObservationJointNameLeftHip,
          VNHumanBodyPoseObservationJointNameLeftKnee,
          VNHumanBodyPoseObservationJointNameLeftAnkle],
        @[VNHumanBodyPoseObservationJointNameRightHip,
          VNHumanBodyPoseObservationJointNameRightKnee,
          VNHumanBodyPoseObservationJointNameRightAnkle]
    ];
    int complete_chains = 0;
    double chain_lengths[4] = {NAN, NAN, NAN, NAN};
    for (int chain_index = 0; chain_index < 4; chain_index++) {
        NSArray<VNHumanBodyPoseObservationJointName> *chain =
            chains[(NSUInteger)chain_index];
        VNRecognizedPoint *a = h3_super_body_point(best_points, chain[0]);
        VNRecognizedPoint *b = h3_super_body_point(best_points, chain[1]);
        VNRecognizedPoint *c = h3_super_body_point(best_points, chain[2]);
        if (a && b && c) {
            complete_chains++;
            chain_lengths[chain_index] =
                h3_super_body_point_distance(a, b) +
                h3_super_body_point_distance(b, c);
        }
    }
    double limb_asymmetry = 1.0;
    for (int pair = 0; pair < 2; pair++) {
        double left = chain_lengths[pair * 2];
        double right = chain_lengths[pair * 2 + 1];
        if (isfinite(left) && isfinite(right) &&
            left > 1e-9 && right > 1e-9) {
            limb_asymmetry = fmax(
                limb_asymmetry, fmax(left / right, right / left));
        }
    }
    metrics->detected = 1;
    metrics->joint_count = best_joint_count;
    metrics->core_joint_count = core_count;
    metrics->complete_chains = complete_chains;
    metrics->joint_coverage =
        (double)best_joint_count / (double)joint_names.count;
    metrics->core_coverage =
        (double)core_count / (double)core_names.count;
    metrics->limb_asymmetry = limb_asymmetry;
    return 1;
}

static CGPoint h3_super_landmark_centroid(
        VNFaceLandmarkRegion2D *region) {
    CGPoint center = CGPointZero;
    if (!region || !region.pointCount) return center;
    const CGPoint *points = region.normalizedPoints;
    for (NSUInteger index = 0; index < region.pointCount; index++) {
        center.x += points[index].x;
        center.y += points[index].y;
    }
    center.x /= (double)region.pointCount;
    center.y /= (double)region.pointCount;
    return center;
}

static NSArray<NSNumber *> *h3_super_normalized_landmark_vector(
        VNFaceObservation *face,
        NSArray<VNFaceLandmarkRegion2D *> *regions) {
    VNFaceLandmarks2D *landmarks = face.landmarks;
    VNFaceLandmarkRegion2D *left_eye = landmarks.leftEye;
    VNFaceLandmarkRegion2D *right_eye = landmarks.rightEye;
    if (!left_eye.pointCount || !right_eye.pointCount) return nil;
    CGPoint left = h3_super_landmark_centroid(left_eye);
    CGPoint right = h3_super_landmark_centroid(right_eye);
    CGPoint center = CGPointMake(
        0.5 * (left.x + right.x), 0.5 * (left.y + right.y));
    double eye_dx = right.x - left.x;
    double eye_dy = right.y - left.y;
    double scale = hypot(eye_dx, eye_dy);
    if (scale < 1e-6) return nil;
    double cosine = eye_dx / scale;
    double sine = eye_dy / scale;
    NSMutableArray<NSNumber *> *vector = [NSMutableArray array];
    for (VNFaceLandmarkRegion2D *region in regions) {
        if (!region.pointCount) return nil;
        const CGPoint *points = region.normalizedPoints;
        for (NSUInteger index = 0; index < region.pointCount; index++) {
            double dx = points[index].x - center.x;
            double dy = points[index].y - center.y;
            [vector addObject:@((cosine * dx + sine * dy) / scale)];
            [vector addObject:@((-sine * dx + cosine * dy) / scale)];
        }
    }
    return vector;
}

static double h3_super_landmark_vector_distance(
        NSArray<NSNumber *> *current,
        NSArray<NSNumber *> *reference) {
    if (!current || !reference || current.count != reference.count ||
        current.count < 2) return NAN;
    double squared = 0.0;
    for (NSUInteger index = 0; index < current.count; index++) {
        double delta = current[index].doubleValue -
                       reference[index].doubleValue;
        squared += delta * delta;
    }
    return sqrt(squared / (0.5 * (double)current.count));
}

int h3_super_check_stage1_face_quality(
        const h3_super_params *params, const h3_result *stage1,
        h3_super_result *result, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!params || !stage1 || !result) {
        return h3_super_fail(error, error_size,
                             "invalid Stage-1 face-quality arguments");
    }
    result->face_quality_gate_applied = 0;
    result->face_quality_gate_passed = 1;
    result->face_quality_sampled_frames = 0;
    result->face_quality_detected_frames = 0;
    result->face_quality_duplicate_frames = 0;
    result->face_quality_detection_rate = 0.0;
    result->face_quality_capture_mean = 0.0;
    result->face_quality_capture_min = 0.0;
    result->face_quality_area_mean = 0.0;
    result->face_quality_area_min = 0.0;
    result->face_quality_feature_mean = 0.0;
    result->face_quality_feature_max = 0.0;
    result->face_quality_landmark_mean = 0.0;
    result->face_quality_landmark_max = 0.0;
    result->face_quality_mouth_landmark_max = 0.0;
    result->face_quality_contour_landmark_max = 0.0;
    result->face_quality_center_jump_mean = 0.0;
    result->face_quality_center_jump_max = 0.0;
    result->face_quality_log_area_jump_mean = 0.0;
    result->face_quality_log_area_jump_max = 0.0;
    result->body_quality_gate_applied = 0;
    result->body_quality_gate_passed = 1;
    result->body_quality_sampled_frames = 0;
    result->body_quality_detected_frames = 0;
    result->body_quality_duplicate_frames = 0;
    result->body_quality_detection_rate = 0.0;
    result->body_quality_joint_coverage_mean = 0.0;
    result->body_quality_joint_coverage_min = 0.0;
    result->body_quality_core_coverage_min = 0.0;
    result->body_quality_complete_chains_mean = 0.0;
    result->body_quality_complete_chains_min = 0.0;
    result->body_quality_limb_asymmetry_max = 0.0;
    int face_gate_enabled =
        params->face_quality_gate || params->person_quality_gate;
    int body_gate_requested = params->person_quality_gate;
    if (!face_gate_enabled) return 1;
    if (!stage1->decoded_rgb || stage1->decoded_width < 1 ||
        stage1->decoded_height < 1 || stage1->decoded_frames < 1) {
        return h3_super_fail(
            error, error_size,
            "Super person-quality gate needs retained Stage-1 RGB frames");
    }

    int stride = params->strict_person_quality ?
        h3_super_strict_person_quality_stride :
        (params->person_quality_gate ?
            h3_super_person_quality_stride : h3_super_face_quality_stride);
    int frames = stage1->decoded_frames;
    if (params->output_frames > 0 && frames > params->output_frames) {
        frames = params->output_frames;
    }
    size_t frame_elements = (size_t)stage1->decoded_width *
                            (size_t)stage1->decoded_height * 3;
    VNFeaturePrintObservation *reference_feature = nil;
    NSArray<NSNumber *> *reference_landmarks = nil;
    NSArray<NSNumber *> *reference_mouth = nil;
    NSArray<NSNumber *> *reference_contour = nil;
    int sampled = 0;
    int detected = 0;
    int duplicates = 0;
    int capture_count = 0;
    int feature_count = 0;
    double capture_sum = 0.0;
    double capture_min = 1.0;
    int area_count = 0;
    double area_sum = 0.0;
    double area_min = 1.0;
    double feature_sum = 0.0;
    double feature_max = 0.0;
    int landmark_count = 0;
    double landmark_sum = 0.0;
    double landmark_max = 0.0;
    double mouth_landmark_max = 0.0;
    double contour_landmark_max = 0.0;
    int previous_face_valid = 0;
    double previous_face_center_x = 0.0;
    double previous_face_center_y = 0.0;
    double previous_face_area = 0.0;
    int center_jump_count = 0;
    double center_jump_sum = 0.0;
    double center_jump_max = 0.0;
    int log_area_jump_count = 0;
    double log_area_jump_sum = 0.0;
    double log_area_jump_max = 0.0;
    int body_gate_enabled = body_gate_requested;
    int body_sampled = 0;
    int body_detected = 0;
    int body_duplicates = 0;
    double body_joint_sum = 0.0;
    double body_joint_min = 1.0;
    double body_core_min = 1.0;
    double body_chains_sum = 0.0;
    double body_chains_min = 4.0;
    double body_limb_asymmetry_max = 1.0;

    for (int frame = 0; frame < frames; frame += stride) {
        @autoreleasepool {
            sampled++;
            CVPixelBufferRef pixels = NULL;
            const float *source = stage1->decoded_rgb +
                (size_t)frame * frame_elements;
            if (!h3_super_face_pixel_buffer(
                    source, stage1->decoded_width, stage1->decoded_height,
                    &pixels, error, error_size)) return 0;
            VNDetectFaceCaptureQualityRequest *quality =
                [[VNDetectFaceCaptureQualityRequest alloc] init];
            VNDetectFaceLandmarksRequest *landmarks =
                [[VNDetectFaceLandmarksRequest alloc] init];
            VNDetectHumanBodyPoseRequest *body = body_gate_enabled ?
                [[VNDetectHumanBodyPoseRequest alloc] init] : nil;
            VNImageRequestHandler *handler =
                [[VNImageRequestHandler alloc]
                    initWithCVPixelBuffer:pixels options:@{}];
            NSError *vision_error = nil;
            NSArray<VNRequest *> *requests = body ?
                @[quality, landmarks, body] : @[quality, landmarks];
            if (![handler performRequests:requests error:&vision_error]) {
                CVPixelBufferRelease(pixels);
                return h3_super_fail(
                    error, error_size,
                    "Apple Vision person detection failed at frame %d: %s",
                    frame, vision_error.localizedDescription.UTF8String);
            }
            NSArray<VNFaceObservation *> *observations = landmarks.results;
            if (!observations.count) observations = quality.results;
            if (observations.count > 1) duplicates++;
            VNFaceObservation *best = nil;
            double best_area = 0.0;
            for (VNFaceObservation *observation in observations) {
                CGRect box = observation.boundingBox;
                double area = box.size.width * box.size.height;
                if (!best || area > best_area) {
                    best = observation;
                    best_area = area;
                }
            }
            h3_super_body_frame_metrics body_metrics;
            memset(&body_metrics, 0, sizeof(body_metrics));
            if (body && !h3_super_body_frame_quality(
                    body.results, &body_metrics, error, error_size)) {
                CVPixelBufferRelease(pixels);
                return 0;
            }
            if (frame == 0 && body_gate_enabled &&
                (!body_metrics.detected ||
                 body_metrics.joint_coverage < 0.85 ||
                 body_metrics.complete_chains < 3)) {
                body_gate_enabled = 0;
                fprintf(stderr,
                        "h3: Stage-1 body quality gate skipped: opening "
                        "frame is a close/partial-body composition "
                        "(detected=%d joints=%.3f chains=%d)\n",
                        body_metrics.detected,
                        body_metrics.joint_coverage,
                        body_metrics.complete_chains);
            }
            if (body_gate_enabled) {
                body_sampled++;
                body_detected += body_metrics.detected;
                body_duplicates += body_metrics.duplicate;
                if (body_metrics.detected) {
                    body_joint_sum += body_metrics.joint_coverage;
                    body_joint_min = fmin(
                        body_joint_min, body_metrics.joint_coverage);
                    body_core_min = fmin(
                        body_core_min, body_metrics.core_coverage);
                    body_chains_sum += body_metrics.complete_chains;
                    body_chains_min = fmin(
                        body_chains_min, body_metrics.complete_chains);
                    body_limb_asymmetry_max = fmax(
                        body_limb_asymmetry_max,
                        body_metrics.limb_asymmetry);
                }
            }
            if (frame == 0 && !best) {
                CVPixelBufferRelease(pixels);
                if (body_gate_requested && body_metrics.detected) {
                    return h3_super_fail(
                        error, error_size,
                        "Stage-1 person quality rejected before LTX: "
                        "opening body is detectable but the face is not");
                }
                fprintf(stderr,
                        "h3: Stage-1 face quality gate skipped: opening "
                        "frame contains no detectable face\n");
                result->face_quality_sampled_frames = 1;
                return 1;
            }
            if (best) {
                detected++;
                area_count++;
                area_sum += best_area;
                area_min = fmin(area_min, best_area);
                CGRect best_box = best.boundingBox;
                double center_x = CGRectGetMidX(best_box);
                double center_y = CGRectGetMidY(best_box);
                if (previous_face_valid && best_area > 0.0 &&
                    previous_face_area > 0.0) {
                    double center_jump = hypot(
                        center_x - previous_face_center_x,
                        center_y - previous_face_center_y);
                    double log_area_jump = fabs(log(
                        best_area / previous_face_area));
                    center_jump_count++;
                    center_jump_sum += center_jump;
                    center_jump_max = fmax(center_jump_max, center_jump);
                    log_area_jump_count++;
                    log_area_jump_sum += log_area_jump;
                    log_area_jump_max = fmax(
                        log_area_jump_max, log_area_jump);
                }
                previous_face_valid = best_area > 0.0;
                previous_face_center_x = center_x;
                previous_face_center_y = center_y;
                previous_face_area = best_area;
                VNFaceObservation *quality_best = nil;
                double quality_best_area = 0.0;
                for (VNFaceObservation *observation in quality.results) {
                    CGRect box = observation.boundingBox;
                    double area = box.size.width * box.size.height;
                    if (!quality_best || area > quality_best_area) {
                        quality_best = observation;
                        quality_best_area = area;
                    }
                }
                if (quality_best.faceCaptureQuality) {
                    double value =
                        quality_best.faceCaptureQuality.doubleValue;
                    capture_sum += value;
                    capture_min = fmin(capture_min, value);
                    capture_count++;
                }
                VNGenerateImageFeaturePrintRequest *feature_request =
                    [[VNGenerateImageFeaturePrintRequest alloc] init];
                feature_request.regionOfInterest = best.boundingBox;
                feature_request.imageCropAndScaleOption =
                    VNImageCropAndScaleOptionScaleFit;
                vision_error = nil;
                if (![handler performRequests:@[feature_request]
                                         error:&vision_error]) {
                    CVPixelBufferRelease(pixels);
                    return h3_super_fail(
                        error, error_size,
                        "Apple Vision face feature print failed at frame %d: %s",
                        frame, vision_error.localizedDescription.UTF8String);
                }
                VNFeaturePrintObservation *feature =
                    feature_request.results.firstObject;
                if (frame == 0 && !feature) {
                    CVPixelBufferRelease(pixels);
                    return h3_super_fail(
                        error, error_size,
                        "Apple Vision returned no opening face feature print");
                }
                if (frame == 0) reference_feature = feature;
                if (feature && reference_feature) {
                    float distance = 0.0f;
                    NSError *distance_error = nil;
                    if (![feature computeDistance:&distance
                         toFeaturePrintObservation:reference_feature
                                               error:&distance_error]) {
                        CVPixelBufferRelease(pixels);
                        return h3_super_fail(
                            error, error_size,
                            "Apple Vision face distance failed at frame %d: %s",
                            frame,
                            distance_error.localizedDescription.UTF8String);
                    }
                    feature_sum += distance;
                    feature_max = fmax(feature_max, distance);
                    feature_count++;
                }
                VNFaceLandmarks2D *face_landmarks = best.landmarks;
                NSArray<NSNumber *> *landmark_vector = nil;
                NSArray<NSNumber *> *mouth_vector = nil;
                NSArray<NSNumber *> *contour_vector = nil;
                if (face_landmarks.leftEye && face_landmarks.rightEye &&
                    face_landmarks.nose && face_landmarks.outerLips &&
                    face_landmarks.faceContour) {
                    landmark_vector = h3_super_normalized_landmark_vector(
                        best, @[
                            face_landmarks.leftEye,
                            face_landmarks.rightEye,
                            face_landmarks.nose,
                            face_landmarks.outerLips,
                            face_landmarks.faceContour
                        ]);
                    mouth_vector = h3_super_normalized_landmark_vector(
                        best, @[face_landmarks.outerLips]);
                    contour_vector = h3_super_normalized_landmark_vector(
                        best, @[face_landmarks.faceContour]);
                }
                if (frame == 0) {
                    reference_landmarks = landmark_vector;
                    reference_mouth = mouth_vector;
                    reference_contour = contour_vector;
                }
                double landmark_distance =
                    h3_super_landmark_vector_distance(
                        landmark_vector, reference_landmarks);
                double mouth_distance =
                    h3_super_landmark_vector_distance(
                        mouth_vector, reference_mouth);
                double contour_distance =
                    h3_super_landmark_vector_distance(
                        contour_vector, reference_contour);
                if (isfinite(landmark_distance)) {
                    landmark_count++;
                    landmark_sum += landmark_distance;
                    landmark_max = fmax(
                        landmark_max, landmark_distance);
                }
                if (isfinite(mouth_distance)) {
                    mouth_landmark_max = fmax(
                        mouth_landmark_max, mouth_distance);
                }
                if (isfinite(contour_distance)) {
                    contour_landmark_max = fmax(
                        contour_landmark_max, contour_distance);
                }
            } else {
                /* Never bridge a temporal jump across a face-detection gap;
                 * the strict detection-rate threshold handles the gap. */
                previous_face_valid = 0;
            }
            CVPixelBufferRelease(pixels);
        }
    }

    result->face_quality_gate_applied = 1;
    result->face_quality_sampled_frames = sampled;
    result->face_quality_detected_frames = detected;
    result->face_quality_duplicate_frames = duplicates;
    result->face_quality_detection_rate = sampled ?
        (double)detected / sampled : 0.0;
    result->face_quality_capture_mean = capture_count ?
        capture_sum / capture_count : 0.0;
    result->face_quality_capture_min = capture_count ? capture_min : 0.0;
    result->face_quality_area_mean = area_count ? area_sum / area_count : 0.0;
    result->face_quality_area_min = area_count ? area_min : 0.0;
    result->face_quality_feature_mean = feature_count ?
        feature_sum / feature_count : 0.0;
    result->face_quality_feature_max = feature_max;
    result->face_quality_landmark_mean = landmark_count ?
        landmark_sum / landmark_count : 0.0;
    result->face_quality_landmark_max = landmark_max;
    result->face_quality_mouth_landmark_max = mouth_landmark_max;
    result->face_quality_contour_landmark_max = contour_landmark_max;
    result->face_quality_center_jump_mean = center_jump_count ?
        center_jump_sum / center_jump_count : 0.0;
    result->face_quality_center_jump_max = center_jump_max;
    result->face_quality_log_area_jump_mean = log_area_jump_count ?
        log_area_jump_sum / log_area_jump_count : 0.0;
    result->face_quality_log_area_jump_max = log_area_jump_max;
    if (body_gate_enabled) {
        result->body_quality_gate_applied = 1;
        result->body_quality_sampled_frames = body_sampled;
        result->body_quality_detected_frames = body_detected;
        result->body_quality_duplicate_frames = body_duplicates;
        result->body_quality_detection_rate = body_sampled ?
            (double)body_detected / body_sampled : 0.0;
        result->body_quality_joint_coverage_mean = body_detected ?
            body_joint_sum / body_detected : 0.0;
        result->body_quality_joint_coverage_min = body_detected ?
            body_joint_min : 0.0;
        result->body_quality_core_coverage_min = body_detected ?
            body_core_min : 0.0;
        result->body_quality_complete_chains_mean = body_detected ?
            body_chains_sum / body_detected : 0.0;
        result->body_quality_complete_chains_min = body_detected ?
            body_chains_min : 0.0;
        result->body_quality_limb_asymmetry_max = body_detected ?
            body_limb_asymmetry_max : 0.0;
    }

    const double minimum_detection_rate = params->strict_person_quality ?
        h3_super_strict_minimum_detection_rate :
        (params->person_quality_gate && params->min_face_area >= 0.006 ?
            h3_super_person_closeup_minimum_detection_rate : 0.80);
    const double minimum_landmark_coverage = 0.90;
    const double maximum_feature_mean = params->person_quality_gate ?
        h3_super_person_maximum_feature_mean :
        h3_super_face_maximum_feature_mean;
    const double maximum_feature_max = 0.95;
    const double maximum_landmark_max = params->person_quality_gate ?
        h3_super_person_maximum_landmark :
        h3_super_face_maximum_landmark;
    const double maximum_mouth_landmark_max = params->person_quality_gate ?
        h3_super_person_maximum_mouth : h3_super_face_maximum_mouth;
    const double maximum_contour_landmark_max = params->person_quality_gate ?
        (params->min_face_area >= 0.006 ?
            h3_super_person_closeup_maximum_contour :
            h3_super_person_maximum_contour) :
        h3_super_face_maximum_contour;
    double landmark_coverage = detected ?
        (double)landmark_count / detected : 0.0;
    double minimum_capture_quality = params->strict_person_quality ?
        h3_super_strict_minimum_capture_quality :
        (params->min_face_area >= 0.006 ?
            h3_super_person_closeup_minimum_capture_quality :
            h3_super_person_minimum_capture_quality);
    int capture_passed = !params->person_quality_gate ||
        (capture_count == detected && capture_count > 0 &&
         result->face_quality_capture_min >=
            minimum_capture_quality);
    int temporal_passed = !params->strict_person_quality ||
        (center_jump_count > 0 && log_area_jump_count > 0 &&
         result->face_quality_center_jump_max <=
            h3_super_strict_maximum_center_jump &&
         result->face_quality_log_area_jump_max <=
            h3_super_strict_maximum_log_area_jump);
    int passed = duplicates == 0 && feature_count > 0 &&
        landmark_coverage >= minimum_landmark_coverage && capture_passed &&
        temporal_passed &&
        result->face_quality_detection_rate >= minimum_detection_rate &&
        result->face_quality_feature_mean <= maximum_feature_mean &&
        result->face_quality_feature_max <= maximum_feature_max &&
        result->face_quality_landmark_max <= maximum_landmark_max &&
        result->face_quality_mouth_landmark_max <=
            maximum_mouth_landmark_max &&
        result->face_quality_contour_landmark_max <=
            maximum_contour_landmark_max &&
        (params->min_face_area <= 0.0 ||
         result->face_quality_area_min >= params->min_face_area);
    result->face_quality_gate_passed = passed;
    fprintf(stderr,
            "h3: Stage-1 face quality gate sampled=%d detected=%d "
            "rate=%.3f duplicates=%d capture=%.3f/%.3f area=%.4f/%.4f "
            "feature_mean=%.3f feature_max=%.3f "
            "landmark_coverage=%.3f landmark_max=%.3f mouth_max=%.3f "
            "contour_max=%.3f center_jump=%.5f/%.5f "
            "log_area_jump=%.5f/%.5f pass=%d\n",
            sampled, detected, result->face_quality_detection_rate,
            duplicates, result->face_quality_capture_mean,
            result->face_quality_capture_min, result->face_quality_area_mean,
            result->face_quality_area_min,
            result->face_quality_feature_mean,
            result->face_quality_feature_max,
            landmark_coverage,
            result->face_quality_landmark_max,
            result->face_quality_mouth_landmark_max,
            result->face_quality_contour_landmark_max,
            result->face_quality_center_jump_mean,
            result->face_quality_center_jump_max,
            result->face_quality_log_area_jump_mean,
            result->face_quality_log_area_jump_max, passed);
    if (!passed && body_gate_enabled) {
        fprintf(stderr,
                "h3: Stage-1 body diagnostic sampled=%d detected=%d "
                "rate=%.3f duplicates=%d joints_mean=%.3f "
                "joints_min=%.3f chains_mean=%.3f chains_min=%.0f "
                "asymmetry_max=%.3f\n",
                body_sampled, body_detected,
                result->body_quality_detection_rate, body_duplicates,
                result->body_quality_joint_coverage_mean,
                result->body_quality_joint_coverage_min,
                result->body_quality_complete_chains_mean,
                result->body_quality_complete_chains_min,
                result->body_quality_limb_asymmetry_max);
    }
    if (!passed) {
        return h3_super_fail(
            error, error_size,
            "Stage-1 face quality rejected before LTX: detection %.3f "
            "(min %.2f), feature mean %.3f (max %.2f), feature max %.3f "
            "(max %.2f), landmark/mouth/contour max %.3f/%.3f/%.3f "
            "(max %.2f/%.2f/%.2f), landmark coverage %.3f (min %.2f), "
            "capture mean/min %.3f/%.3f (min %.2f), "
            "face area mean/min %.4f/%.4f (min %.4f), center jump "
            "mean/max %.5f/%.5f (max %.2f), log-area jump mean/max "
            "%.5f/%.5f (max %.2f), "
            "duplicate frames %d",
            result->face_quality_detection_rate, minimum_detection_rate,
            result->face_quality_feature_mean, maximum_feature_mean,
            result->face_quality_feature_max, maximum_feature_max,
            result->face_quality_landmark_max,
            result->face_quality_mouth_landmark_max,
            result->face_quality_contour_landmark_max,
            maximum_landmark_max,
            maximum_mouth_landmark_max,
            maximum_contour_landmark_max,
            landmark_coverage, minimum_landmark_coverage,
            result->face_quality_capture_mean,
            result->face_quality_capture_min,
            params->person_quality_gate ? minimum_capture_quality : 0.0,
            result->face_quality_area_mean,
            result->face_quality_area_min,
            params->min_face_area,
            result->face_quality_center_jump_mean,
            result->face_quality_center_jump_max,
            params->strict_person_quality ?
                h3_super_strict_maximum_center_jump : 0.0,
            result->face_quality_log_area_jump_mean,
            result->face_quality_log_area_jump_max,
            params->strict_person_quality ?
                h3_super_strict_maximum_log_area_jump : 0.0,
            duplicates);
    }
    if (body_gate_enabled) {
        const double body_minimum_detection_rate = 0.95;
        const double body_minimum_joint_coverage_mean = 0.95;
        const double body_minimum_joint_coverage_min = 0.85;
        const double body_minimum_core_coverage_min = 0.80;
        const double body_minimum_complete_chains_mean = 3.50;
        const double body_minimum_complete_chains_min = 2.0;
        const double body_maximum_limb_asymmetry = 2.25;
        int body_passed = body_duplicates == 0 && body_detected > 0 &&
            result->body_quality_detection_rate >=
                body_minimum_detection_rate &&
            result->body_quality_joint_coverage_mean >=
                body_minimum_joint_coverage_mean &&
            result->body_quality_joint_coverage_min >=
                body_minimum_joint_coverage_min &&
            result->body_quality_core_coverage_min >=
                body_minimum_core_coverage_min &&
            result->body_quality_complete_chains_mean >=
                body_minimum_complete_chains_mean &&
            result->body_quality_complete_chains_min >=
                body_minimum_complete_chains_min &&
            result->body_quality_limb_asymmetry_max <=
                body_maximum_limb_asymmetry;
        result->body_quality_gate_passed = body_passed;
        fprintf(stderr,
                "h3: Stage-1 body quality gate sampled=%d detected=%d "
                "rate=%.3f duplicates=%d joints_mean=%.3f "
                "joints_min=%.3f chains_mean=%.3f chains_min=%.0f "
                "asymmetry_max=%.3f pass=%d\n",
                body_sampled, body_detected,
                result->body_quality_detection_rate, body_duplicates,
                result->body_quality_joint_coverage_mean,
                result->body_quality_joint_coverage_min,
                result->body_quality_complete_chains_mean,
                result->body_quality_complete_chains_min,
                result->body_quality_limb_asymmetry_max, body_passed);
        if (!body_passed) {
            return h3_super_fail(
                error, error_size,
                "Stage-1 body quality rejected before LTX: detection %.3f "
                "(min %.2f), joints mean/min %.3f/%.3f "
                "(min %.2f/%.2f), core min %.3f (min %.2f), "
                "chains mean/min %.3f/%.0f (min %.2f/%.0f), "
                "limb asymmetry %.3f (max %.2f), duplicate frames %d",
                result->body_quality_detection_rate,
                body_minimum_detection_rate,
                result->body_quality_joint_coverage_mean,
                result->body_quality_joint_coverage_min,
                body_minimum_joint_coverage_mean,
                body_minimum_joint_coverage_min,
                result->body_quality_core_coverage_min,
                body_minimum_core_coverage_min,
                result->body_quality_complete_chains_mean,
                result->body_quality_complete_chains_min,
                body_minimum_complete_chains_mean,
                body_minimum_complete_chains_min,
                result->body_quality_limb_asymmetry_max,
                body_maximum_limb_asymmetry, body_duplicates);
        }
    }
    return 1;
}

int h3_super_check_output_person_quality(
        const h3_super_params *params, const char *output_path,
        h3_super_result *result, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!params || !output_path || !*output_path || !result) {
        return h3_super_fail(error, error_size,
                             "invalid refined-output quality arguments");
    }
    result->output_quality_gate_applied = 0;
    result->output_quality_gate_passed = 1;
    result->output_face_sampled_frames = 0;
    result->output_face_detected_frames = 0;
    result->output_face_duplicate_frames = 0;
    result->output_face_detection_rate = 0.0;
    result->output_face_capture_mean = 0.0;
    result->output_face_capture_min = 0.0;
    result->output_face_area_mean = 0.0;
    result->output_face_area_min = 0.0;
    result->output_face_feature_mean = 0.0;
    result->output_face_feature_max = 0.0;
    result->output_face_landmark_max = 0.0;
    result->output_face_mouth_landmark_max = 0.0;
    result->output_face_contour_landmark_max = 0.0;
    result->output_face_center_jump_mean = 0.0;
    result->output_face_center_jump_max = 0.0;
    result->output_face_log_area_jump_mean = 0.0;
    result->output_face_log_area_jump_max = 0.0;
    result->output_body_gate_applied = 0;
    result->output_body_gate_passed = 1;
    result->output_body_sampled_frames = 0;
    result->output_body_detected_frames = 0;
    result->output_body_duplicate_frames = 0;
    result->output_body_detection_rate = 0.0;
    result->output_body_joint_coverage_mean = 0.0;
    result->output_body_joint_coverage_min = 0.0;
    result->output_body_core_coverage_min = 0.0;
    result->output_body_complete_chains_mean = 0.0;
    result->output_body_complete_chains_min = 0.0;
    result->output_body_limb_asymmetry_max = 0.0;
    int output_gate_requested = params->person_quality_gate ||
        params->output_quality_gate;
    if (!output_gate_requested) return 1;
    if (params->person_quality_gate &&
        !params->output_quality_gate &&
        !result->face_quality_gate_applied) return 1;

    @autoreleasepool {
        NSString *path = [[NSString stringWithUTF8String:output_path]
            stringByStandardizingPath];
        if (!path) {
            return h3_super_fail(error, error_size,
                                 "cannot encode refined output path");
        }
        AVURLAsset *asset = [AVURLAsset
            URLAssetWithURL:[NSURL fileURLWithPath:path] options:nil];
        __block NSArray<AVAssetTrack *> *tracks = nil;
        __block NSError *track_error = nil;
        dispatch_semaphore_t loaded = dispatch_semaphore_create(0);
        [asset loadTracksWithMediaType:AVMediaTypeVideo
                     completionHandler:^(NSArray<AVAssetTrack *> *value,
                                         NSError *load_error) {
            tracks = value;
            track_error = load_error;
            dispatch_semaphore_signal(loaded);
        }];
        dispatch_semaphore_wait(loaded, DISPATCH_TIME_FOREVER);
        if (track_error || !tracks.count) {
            return h3_super_fail(
                error, error_size,
                "cannot load refined output video track: %s",
                track_error ? track_error.localizedDescription.UTF8String :
                    output_path);
        }

        NSError *reader_error = nil;
        AVAssetReader *reader = [[AVAssetReader alloc]
            initWithAsset:asset error:&reader_error];
        if (!reader) {
            return h3_super_fail(
                error, error_size, "cannot open refined output: %s",
                reader_error.localizedDescription.UTF8String);
        }
        NSDictionary *settings = @{
            (id)kCVPixelBufferPixelFormatTypeKey:
                @(kCVPixelFormatType_32BGRA)
        };
        AVAssetReaderTrackOutput *video = [[AVAssetReaderTrackOutput alloc]
            initWithTrack:tracks.firstObject outputSettings:settings];
        video.alwaysCopiesSampleData = NO;
        if (![reader canAddOutput:video]) {
            return h3_super_fail(error, error_size,
                                 "cannot configure refined output reader");
        }
        [reader addOutput:video];
        if (![reader startReading]) {
            return h3_super_fail(
                error, error_size, "cannot start refined output reader: %s",
                reader.error.localizedDescription.UTF8String);
        }

        int stride = params->strict_person_quality ?
            h3_super_strict_person_quality_stride :
            h3_super_person_quality_stride;
        int frame_index = 0;
        int sampled = 0;
        int detected = 0;
        int duplicates = 0;
        int feature_count = 0;
        double feature_sum = 0.0;
        double feature_max = 0.0;
        int capture_count = 0;
        double capture_sum = 0.0;
        double capture_min = 1.0;
        int area_count = 0;
        double area_sum = 0.0;
        double area_min = 1.0;
        int landmark_count = 0;
        double landmark_max = 0.0;
        double mouth_max = 0.0;
        double contour_max = 0.0;
        int previous_face_valid = 0;
        double previous_face_center_x = 0.0;
        double previous_face_center_y = 0.0;
        double previous_face_area = 0.0;
        int center_jump_count = 0;
        double center_jump_sum = 0.0;
        double center_jump_max = 0.0;
        int log_area_jump_count = 0;
        double log_area_jump_sum = 0.0;
        double log_area_jump_max = 0.0;
        VNFeaturePrintObservation *reference_feature = nil;
        NSArray<NSNumber *> *reference_landmarks = nil;
        NSArray<NSNumber *> *reference_mouth = nil;
        NSArray<NSNumber *> *reference_contour = nil;
        int body_enabled = params->output_quality_gate ? 1 :
            result->body_quality_gate_applied;
        int body_sampled = 0;
        int body_detected = 0;
        int body_duplicates = 0;
        double body_joint_sum = 0.0;
        double body_joint_min = 1.0;
        double body_core_min = 1.0;
        double body_chains_sum = 0.0;
        double body_chains_min = 4.0;
        double body_asymmetry_max = 1.0;

        while (reader.status == AVAssetReaderStatusReading) {
            CMSampleBufferRef sample = [video copyNextSampleBuffer];
            if (!sample) break;
            if (frame_index % stride == 0) {
                @autoreleasepool {
                    sampled++;
                    CVPixelBufferRef pixels =
                        CMSampleBufferGetImageBuffer(sample);
                    VNDetectFaceCaptureQualityRequest *quality =
                        [[VNDetectFaceCaptureQualityRequest alloc] init];
                    VNDetectFaceLandmarksRequest *landmarks =
                        [[VNDetectFaceLandmarksRequest alloc] init];
                    VNDetectHumanBodyPoseRequest *body = body_enabled ?
                        [[VNDetectHumanBodyPoseRequest alloc] init] : nil;
                    VNImageRequestHandler *handler =
                        [[VNImageRequestHandler alloc]
                            initWithCVPixelBuffer:pixels options:@{}];
                    NSError *vision_error = nil;
                    NSArray<VNRequest *> *requests = body ?
                        @[quality, landmarks, body] : @[quality, landmarks];
                    if (![handler performRequests:requests
                                             error:&vision_error]) {
                        CFRelease(sample);
                        return h3_super_fail(
                            error, error_size,
                            "Apple Vision refined-output analysis failed at "
                            "frame %d: %s", frame_index,
                            vision_error.localizedDescription.UTF8String);
                    }
                    NSArray<VNFaceObservation *> *observations =
                        landmarks.results;
                    if (!observations.count) observations = quality.results;
                    if (observations.count > 1) duplicates++;
                    VNFaceObservation *best = nil;
                    double best_area = 0.0;
                    for (VNFaceObservation *observation in observations) {
                        CGRect box = observation.boundingBox;
                        double area = box.size.width * box.size.height;
                        if (!best || area > best_area) {
                            best = observation;
                            best_area = area;
                        }
                    }
                    if (frame_index == 0 && !best) {
                        CFRelease(sample);
                        return h3_super_fail(
                            error, error_size,
                            "refined output lost the opening face");
                    }
                    if (best) {
                        detected++;
                        area_count++;
                        area_sum += best_area;
                        area_min = fmin(area_min, best_area);
                        CGRect best_box = best.boundingBox;
                        double center_x = CGRectGetMidX(best_box);
                        double center_y = CGRectGetMidY(best_box);
                        if (previous_face_valid && best_area > 0.0 &&
                            previous_face_area > 0.0) {
                            double center_jump = hypot(
                                center_x - previous_face_center_x,
                                center_y - previous_face_center_y);
                            double log_area_jump = fabs(log(
                                best_area / previous_face_area));
                            center_jump_count++;
                            center_jump_sum += center_jump;
                            center_jump_max = fmax(
                                center_jump_max, center_jump);
                            log_area_jump_count++;
                            log_area_jump_sum += log_area_jump;
                            log_area_jump_max = fmax(
                                log_area_jump_max, log_area_jump);
                        }
                        previous_face_valid = best_area > 0.0;
                        previous_face_center_x = center_x;
                        previous_face_center_y = center_y;
                        previous_face_area = best_area;
                        VNFaceObservation *quality_best = nil;
                        double quality_best_area = 0.0;
                        for (VNFaceObservation *observation in quality.results) {
                            CGRect quality_box = observation.boundingBox;
                            double quality_area = quality_box.size.width *
                                quality_box.size.height;
                            if (!quality_best ||
                                quality_area > quality_best_area) {
                                quality_best = observation;
                                quality_best_area = quality_area;
                            }
                        }
                        if (quality_best.faceCaptureQuality) {
                            double capture =
                                quality_best.faceCaptureQuality.doubleValue;
                            capture_count++;
                            capture_sum += capture;
                            capture_min = fmin(capture_min, capture);
                        }
                        VNGenerateImageFeaturePrintRequest *feature_request =
                            [[VNGenerateImageFeaturePrintRequest alloc] init];
                        feature_request.regionOfInterest = best.boundingBox;
                        feature_request.imageCropAndScaleOption =
                            VNImageCropAndScaleOptionScaleFit;
                        if (![handler performRequests:@[feature_request]
                                                 error:&vision_error]) {
                            CFRelease(sample);
                            return h3_super_fail(
                                error, error_size,
                                "Apple Vision refined-output feature print "
                                "failed at frame %d: %s", frame_index,
                                vision_error.localizedDescription.UTF8String);
                        }
                        VNFeaturePrintObservation *feature =
                            feature_request.results.firstObject;
                        if (frame_index == 0) reference_feature = feature;
                        if (feature && reference_feature) {
                            float distance = 0.0f;
                            NSError *distance_error = nil;
                            if (![feature computeDistance:&distance
                             toFeaturePrintObservation:reference_feature
                                                   error:&distance_error]) {
                                CFRelease(sample);
                                return h3_super_fail(
                                    error, error_size,
                                    "Apple Vision refined-output face "
                                    "distance failed at frame %d: %s",
                                    frame_index,
                                    distance_error.localizedDescription.UTF8String);
                            }
                            feature_count++;
                            feature_sum += distance;
                            feature_max = fmax(feature_max, distance);
                        }
                        VNFaceLandmarks2D *face = best.landmarks;
                        NSArray<NSNumber *> *landmark_vector = nil;
                        NSArray<NSNumber *> *mouth_vector = nil;
                        NSArray<NSNumber *> *contour_vector = nil;
                        if (face.leftEye && face.rightEye && face.nose &&
                            face.outerLips && face.faceContour) {
                            landmark_vector =
                                h3_super_normalized_landmark_vector(
                                    best, @[face.leftEye, face.rightEye,
                                            face.nose, face.outerLips,
                                            face.faceContour]);
                            mouth_vector =
                                h3_super_normalized_landmark_vector(
                                    best, @[face.outerLips]);
                            contour_vector =
                                h3_super_normalized_landmark_vector(
                                    best, @[face.faceContour]);
                        }
                        if (frame_index == 0) {
                            reference_landmarks = landmark_vector;
                            reference_mouth = mouth_vector;
                            reference_contour = contour_vector;
                        }
                        double landmark_distance =
                            h3_super_landmark_vector_distance(
                                landmark_vector, reference_landmarks);
                        double mouth_distance =
                            h3_super_landmark_vector_distance(
                                mouth_vector, reference_mouth);
                        double contour_distance =
                            h3_super_landmark_vector_distance(
                                contour_vector, reference_contour);
                        if (isfinite(landmark_distance)) {
                            landmark_count++;
                            landmark_max = fmax(
                                landmark_max, landmark_distance);
                        }
                        if (isfinite(mouth_distance))
                            mouth_max = fmax(mouth_max, mouth_distance);
                        if (isfinite(contour_distance))
                            contour_max = fmax(contour_max, contour_distance);
                    } else {
                        previous_face_valid = 0;
                    }
                    if (body) {
                        h3_super_body_frame_metrics metrics;
                        if (!h3_super_body_frame_quality(
                                body.results, &metrics,
                                error, error_size)) {
                            CFRelease(sample);
                            return 0;
                        }
                        if (frame_index == 0 &&
                            (!metrics.detected ||
                             metrics.joint_coverage < 0.85 ||
                             metrics.complete_chains < 3)) {
                            body_enabled = 0;
                            fprintf(stderr,
                                    "h3: refined-output body gate skipped: "
                                    "opening is a close/partial-body "
                                    "composition (detected=%d joints=%.3f "
                                    "chains=%d)\n",
                                    metrics.detected,
                                    metrics.joint_coverage,
                                    metrics.complete_chains);
                        } else if (body_enabled) {
                            body_sampled++;
                            body_detected += metrics.detected;
                            body_duplicates += metrics.duplicate;
                        }
                        if (body_enabled && metrics.detected) {
                            body_joint_sum += metrics.joint_coverage;
                            body_joint_min = fmin(
                                body_joint_min, metrics.joint_coverage);
                            body_core_min = fmin(
                                body_core_min, metrics.core_coverage);
                            body_chains_sum += metrics.complete_chains;
                            body_chains_min = fmin(
                                body_chains_min, metrics.complete_chains);
                            body_asymmetry_max = fmax(
                                body_asymmetry_max,
                                metrics.limb_asymmetry);
                        }
                    }
                }
            }
            CFRelease(sample);
            frame_index++;
        }
        if (reader.status == AVAssetReaderStatusFailed) {
            return h3_super_fail(
                error, error_size, "refined output decode failed: %s",
                reader.error.localizedDescription.UTF8String);
        }

        result->output_quality_gate_applied = 1;
        result->output_face_sampled_frames = sampled;
        result->output_face_detected_frames = detected;
        result->output_face_duplicate_frames = duplicates;
        result->output_face_detection_rate = sampled ?
            (double)detected / sampled : 0.0;
        result->output_face_capture_mean = capture_count ?
            capture_sum / capture_count : 0.0;
        result->output_face_capture_min = capture_count ? capture_min : 0.0;
        result->output_face_area_mean = area_count ?
            area_sum / area_count : 0.0;
        result->output_face_area_min = area_count ? area_min : 0.0;
        result->output_face_feature_mean = feature_count ?
            feature_sum / feature_count : 0.0;
        result->output_face_feature_max = feature_max;
        result->output_face_landmark_max = landmark_max;
        result->output_face_mouth_landmark_max = mouth_max;
        result->output_face_contour_landmark_max = contour_max;
        result->output_face_center_jump_mean = center_jump_count ?
            center_jump_sum / center_jump_count : 0.0;
        result->output_face_center_jump_max = center_jump_max;
        result->output_face_log_area_jump_mean = log_area_jump_count ?
            log_area_jump_sum / log_area_jump_count : 0.0;
        result->output_face_log_area_jump_max = log_area_jump_max;
        if (body_enabled) {
            result->output_body_gate_applied = 1;
            result->output_body_sampled_frames = body_sampled;
            result->output_body_detected_frames = body_detected;
            result->output_body_duplicate_frames = body_duplicates;
            result->output_body_detection_rate = body_sampled ?
                (double)body_detected / body_sampled : 0.0;
            result->output_body_joint_coverage_mean = body_detected ?
                body_joint_sum / body_detected : 0.0;
            result->output_body_joint_coverage_min = body_detected ?
                body_joint_min : 0.0;
            result->output_body_core_coverage_min = body_detected ?
                body_core_min : 0.0;
            result->output_body_complete_chains_mean = body_detected ?
                body_chains_sum / body_detected : 0.0;
            result->output_body_complete_chains_min = body_detected ?
                body_chains_min : 0.0;
            result->output_body_limb_asymmetry_max = body_detected ?
                body_asymmetry_max : 0.0;
        }

        const double minimum_detection = params->strict_person_quality ?
            h3_super_strict_minimum_detection_rate :
            (params->min_face_area >= 0.006 ?
                h3_super_person_closeup_minimum_detection_rate : 0.85);
        const double maximum_feature_mean = 0.65;
        const double maximum_feature_max = 0.95;
        const double maximum_landmark = 0.60;
        const double maximum_mouth = 0.60;
        const double maximum_contour = params->min_face_area >= 0.006 ?
            h3_super_output_closeup_maximum_contour : 0.70;
        const double minimum_landmark_coverage = 0.90;
        const double minimum_capture_quality =
            params->strict_person_quality ?
                h3_super_strict_minimum_capture_quality :
                (params->min_face_area >= 0.006 ?
                    h3_super_person_closeup_minimum_capture_quality :
                    h3_super_person_minimum_capture_quality);
        double landmark_coverage = detected ?
            (double)landmark_count / detected : 0.0;
        int temporal_passed = !params->strict_person_quality ||
            (center_jump_count > 0 && log_area_jump_count > 0 &&
             result->output_face_center_jump_max <=
                h3_super_strict_maximum_center_jump &&
             result->output_face_log_area_jump_max <=
                h3_super_strict_maximum_log_area_jump);
        int face_passed = sampled > 0 && detected > 0 && duplicates == 0 &&
            feature_count > 0 &&
            landmark_coverage >= minimum_landmark_coverage &&
            temporal_passed &&
            capture_count == detected &&
            result->output_face_capture_min >=
                minimum_capture_quality &&
            result->output_face_detection_rate >= minimum_detection &&
            result->output_face_feature_mean <= maximum_feature_mean &&
            result->output_face_feature_max <= maximum_feature_max &&
            landmark_max <= maximum_landmark && mouth_max <= maximum_mouth &&
            contour_max <= maximum_contour &&
            (params->min_face_area <= 0.0 ||
             result->output_face_area_min >= params->min_face_area);
        int body_passed = 1;
        if (body_enabled) {
            body_passed = body_sampled > 0 && body_detected > 0 &&
                body_duplicates == 0 &&
                result->output_body_detection_rate >= 0.95 &&
                result->output_body_joint_coverage_mean >= 0.95 &&
                result->output_body_joint_coverage_min >= 0.85 &&
                result->output_body_core_coverage_min >= 0.80 &&
                result->output_body_complete_chains_mean >= 3.50 &&
                result->output_body_complete_chains_min >= 2.0 &&
                result->output_body_limb_asymmetry_max <= 2.25;
            result->output_body_gate_passed = body_passed;
        }
        int passed = face_passed && body_passed;
        result->output_quality_gate_passed = passed;
        fprintf(stderr,
                "h3: refined-output person gate face detected=%d/%d "
                "duplicates=%d capture=%.3f/%.3f area=%.4f/%.4f "
                "feature=%.3f/%.3f landmark_coverage=%.3f "
                "landmark/mouth/contour=%.3f/%.3f/%.3f "
                "center_jump=%.5f/%.5f log_area_jump=%.5f/%.5f; "
                "body detected=%d/%d duplicates=%d joints=%.3f/%.3f "
                "chains=%.3f/%.0f asymmetry=%.3f pass=%d\n",
                detected, sampled, duplicates,
                result->output_face_capture_mean,
                result->output_face_capture_min,
                result->output_face_area_mean,
                result->output_face_area_min,
                result->output_face_feature_mean,
                result->output_face_feature_max, landmark_coverage,
                landmark_max, mouth_max, contour_max,
                result->output_face_center_jump_mean,
                result->output_face_center_jump_max,
                result->output_face_log_area_jump_mean,
                result->output_face_log_area_jump_max,
                body_detected, body_sampled, body_duplicates,
                result->output_body_joint_coverage_mean,
                result->output_body_joint_coverage_min,
                result->output_body_complete_chains_mean,
                result->output_body_complete_chains_min,
                result->output_body_limb_asymmetry_max, passed);
        if (!passed) {
            return h3_super_fail(
                error, error_size,
                "refined output person quality rejected: face detection "
                "%.3f (min %.2f), feature %.3f/%.3f "
                "(max %.2f/%.2f), landmark coverage %.3f (min %.2f), "
                "capture %.3f/%.3f (min %.2f), landmark/mouth/contour "
                "%.3f/%.3f/%.3f (max %.2f/%.2f/%.2f), body detection "
                "%.3f, joints %.3f/%.3f, chains %.3f/%.0f, "
                "asymmetry %.3f, face area mean/min %.4f/%.4f "
                "(min %.4f), center jump mean/max %.5f/%.5f (max %.2f), "
                "log-area jump mean/max %.5f/%.5f (max %.2f), "
                "duplicate face/body %d/%d",
                result->output_face_detection_rate, minimum_detection,
                result->output_face_feature_mean,
                result->output_face_feature_max, maximum_feature_mean,
                maximum_feature_max, landmark_coverage,
                minimum_landmark_coverage,
                result->output_face_capture_mean,
                result->output_face_capture_min,
                minimum_capture_quality,
                landmark_max, mouth_max, contour_max,
                maximum_landmark, maximum_mouth, maximum_contour,
                result->output_body_detection_rate,
                result->output_body_joint_coverage_mean,
                result->output_body_joint_coverage_min,
                result->output_body_complete_chains_mean,
                result->output_body_complete_chains_min,
                result->output_body_limb_asymmetry_max,
                result->output_face_area_mean,
                result->output_face_area_min,
                params->min_face_area,
                result->output_face_center_jump_mean,
                result->output_face_center_jump_max,
                params->strict_person_quality ?
                    h3_super_strict_maximum_center_jump : 0.0,
                result->output_face_log_area_jump_mean,
                result->output_face_log_area_jump_max,
                params->strict_person_quality ?
                    h3_super_strict_maximum_log_area_jump : 0.0,
                duplicates,
                body_duplicates);
        }
    }
    return 1;
}

static NSString *h3_super_string(const char *value) {
    if (!value) return nil;
    return [NSString stringWithUTF8String:value];
}

static int h3_super_copy_string(char *destination, size_t destination_size,
                                NSString *source, char *error,
                                size_t error_size) {
    const char *utf8 = source.UTF8String;
    if (!utf8) return h3_super_fail(error, error_size,
                                    "cannot encode path as UTF-8");
    int length = snprintf(destination, destination_size, "%s", utf8);
    if (length < 0 || (size_t)length >= destination_size) {
        return h3_super_fail(error, error_size, "path is too long: %s", utf8);
    }
    return 1;
}

static NSData *h3_super_json_data(id object, char *error,
                                  size_t error_size) {
    NSError *json_error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:object
                                                   options:0
                                                     error:&json_error];
    if (!data) {
        h3_super_fail(error, error_size, "cannot encode JSON: %s",
                      json_error.localizedDescription.UTF8String);
    }
    return data;
}

static id h3_super_json_object(NSData *data, char *error,
                               size_t error_size) {
    if (!data.length) {
        h3_super_fail(error, error_size, "backend returned an empty response");
        return nil;
    }
    NSError *json_error = nil;
    id object = [NSJSONSerialization JSONObjectWithData:data
                                                options:0
                                                  error:&json_error];
    if (!object) {
        h3_super_fail(error, error_size, "backend returned invalid JSON: %s",
                      json_error.localizedDescription.UTF8String);
    }
    return object;
}

static NSData *h3_super_http(const h3_super_params *params,
                             NSString *method, NSString *path,
                             NSData *body, NSInteger *status,
                             char *error, size_t error_size) {
    NSString *base = h3_super_string(params->endpoint);
    if (!base.length) {
        h3_super_fail(error, error_size, "Super endpoint is empty");
        return nil;
    }
    while ([base hasSuffix:@"/"]) base = [base substringToIndex:base.length - 1];
    NSURL *url = [NSURL URLWithString:[base stringByAppendingString:path]];
    if (!url) {
        h3_super_fail(error, error_size, "invalid Super endpoint: %s",
                      params->endpoint);
        return nil;
    }
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = method;
    request.timeoutInterval = params->timeout_seconds;
    if (body) {
        request.HTTPBody = body;
        [request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    }

    NSURLSessionConfiguration *configuration =
        NSURLSessionConfiguration.ephemeralSessionConfiguration;
    configuration.timeoutIntervalForRequest = params->timeout_seconds;
    configuration.timeoutIntervalForResource = params->timeout_seconds;
    NSURLSession *session = [NSURLSession sessionWithConfiguration:configuration];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block NSData *data = nil;
    __block NSURLResponse *response = nil;
    __block NSError *request_error = nil;
    NSURLSessionDataTask *task = [session dataTaskWithRequest:request
        completionHandler:^(NSData *task_data, NSURLResponse *task_response,
                            NSError *task_error) {
            data = task_data;
            response = task_response;
            request_error = task_error;
            dispatch_semaphore_signal(completed);
        }];
    [task resume];
    double wait_seconds = params->timeout_seconds + 5.0;
    int64_t wait_nanoseconds = (int64_t)(wait_seconds * 1000000000.0);
    long wait_status = dispatch_semaphore_wait(
        completed, dispatch_time(DISPATCH_TIME_NOW, wait_nanoseconds));
    if (wait_status != 0) {
        [task cancel];
        [session invalidateAndCancel];
        h3_super_fail(error, error_size,
                      "backend request %s timed out after %.0f seconds",
                      path.UTF8String, params->timeout_seconds);
        return nil;
    }
    [session finishTasksAndInvalidate];
    if (!data) {
        h3_super_fail(error, error_size, "backend request %s failed: %s",
                      path.UTF8String,
                      request_error.localizedDescription.UTF8String);
        return nil;
    }
    NSInteger code = [(NSHTTPURLResponse *)response statusCode];
    if (status) *status = code;
    if (code < 200 || code >= 300) {
        NSString *message = [[NSString alloc] initWithData:data
                                                   encoding:NSUTF8StringEncoding];
        h3_super_fail(error, error_size,
                      "backend request %s returned HTTP %ld: %.800s",
                      path.UTF8String, (long)code,
                      message ? message.UTF8String : "non-text response");
        return nil;
    }
    return data;
}

static int h3_super_directory(const char *path, int writable,
                              char *error, size_t error_size) {
    if (!path || !*path) return h3_super_fail(error, error_size,
                                              "required directory is empty");
    struct stat info;
    if (stat(path, &info) != 0 || !S_ISDIR(info.st_mode)) {
        return h3_super_fail(error, error_size,
                             "required directory is missing: %s", path);
    }
    if (writable && access(path, W_OK) != 0) {
        return h3_super_fail(error, error_size,
                             "directory is not writable: %s", path);
    }
    return 1;
}

static int h3_super_sha256_file(const char *path, char digest_hex[65],
                                uint64_t *bytes, char *error,
                                size_t error_size) {
    if (!path || !*path || !digest_hex) {
        return h3_super_fail(error, error_size,
                             "invalid SHA-256 file arguments");
    }
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        return h3_super_fail(error, error_size, "cannot open %s: %s",
                             path, strerror(errno));
    }
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) {
        close(descriptor);
        return h3_super_fail(error, error_size,
                             "cannot initialize SHA-256 for %s", path);
    }
    unsigned char buffer[1024 * 1024];
    uint64_t total = 0;
    int ok = 1;
    for (;;) {
        ssize_t got = read(descriptor, buffer, sizeof(buffer));
        if (got == 0) break;
        if (got < 0) {
            if (errno == EINTR) continue;
            ok = h3_super_fail(error, error_size, "cannot read %s: %s",
                               path, strerror(errno));
            break;
        }
        if (CC_SHA256_Update(&context, buffer, (CC_LONG)got) != 1) {
            ok = h3_super_fail(error, error_size,
                               "cannot update SHA-256 for %s", path);
            break;
        }
        total += (uint64_t)got;
    }
    if (close(descriptor) != 0 && ok) {
        ok = h3_super_fail(error, error_size, "cannot close %s: %s",
                           path, strerror(errno));
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (ok && CC_SHA256_Final(digest, &context) != 1) {
        ok = h3_super_fail(error, error_size,
                           "cannot finish SHA-256 for %s", path);
    }
    if (!ok) return 0;
    for (size_t index = 0; index < sizeof(digest); index++) {
        snprintf(digest_hex + index * 2, 3, "%02x", digest[index]);
    }
    digest_hex[64] = '\0';
    if (bytes) *bytes = total;
    return 1;
}

static int h3_super_object_contains(id object, NSString *needle) {
    if ([object isKindOfClass:[NSString class]]) {
        return [(NSString *)object isEqualToString:needle];
    }
    if ([object isKindOfClass:[NSArray class]]) {
        for (id value in (NSArray *)object) {
            if (h3_super_object_contains(value, needle)) return 1;
        }
    } else if ([object isKindOfClass:[NSDictionary class]]) {
        for (id key in (NSDictionary *)object) {
            if (h3_super_object_contains(key, needle) ||
                h3_super_object_contains([(NSDictionary *)object objectForKey:key],
                                         needle)) return 1;
        }
    }
    return 0;
}

static int h3_super_swapouts(uint64_t *swapouts) {
    vm_statistics64_data_t statistics;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t status = host_statistics64(
        mach_host_self(), HOST_VM_INFO64, (host_info64_t)&statistics, &count);
    if (status != KERN_SUCCESS) return 0;
    if (swapouts) *swapouts = (uint64_t)statistics.swapouts;
    return 1;
}

static int h3_super_system_memory(const h3_super_params *params,
                                  uint64_t *total, uint64_t *free_bytes) {
    char ignored[256];
    NSInteger status = 0;
    NSData *data = h3_super_http(params, @"GET", @"/system_stats", nil,
                                 &status, ignored, sizeof(ignored));
    if (!data) return 0;
    id object = h3_super_json_object(data, ignored, sizeof(ignored));
    NSDictionary *system = [object isKindOfClass:[NSDictionary class]] ?
        [object objectForKey:@"system"] : nil;
    NSNumber *ram_total = [system isKindOfClass:[NSDictionary class]] ?
        system[@"ram_total"] : nil;
    NSNumber *ram_free = [system isKindOfClass:[NSDictionary class]] ?
        system[@"ram_free"] : nil;
    if (![ram_total isKindOfClass:[NSNumber class]] ||
        ![ram_free isKindOfClass:[NSNumber class]]) return 0;
    if (total) *total = ram_total.unsignedLongLongValue;
    if (free_bytes) *free_bytes = ram_free.unsignedLongLongValue;
    return 1;
}

static int h3_super_release_backend(const h3_super_params *params,
                                    char *error, size_t error_size) {
    NSDictionary *request = @{
        @"unload_models": @YES,
        @"free_memory": @YES
    };
    NSData *body = h3_super_json_data(request, error, error_size);
    if (!body) return 0;
    NSInteger status = 0;
    if (!h3_super_http(params, @"POST", @"/free", body, &status,
                       error, error_size)) return 0;
    /* /free wakes Comfy's worker and resets its executor asynchronously. Give
     * it a head start so an immediately queued phase cannot win the race. If
     * cleanup takes longer, the single worker still serializes the next job. */
    struct timespec settle = {0, 250000000L};
    nanosleep(&settle, NULL);
    return 1;
}

void h3_super_params_init(h3_super_params *params) {
    if (!params) return;
    memset(params, 0, sizeof(*params));
    h3_super_params_set_profile(params, H3_SUPER_PROFILE_480P);
    params->endpoint = "http://127.0.0.1:8188";
    const char *input_dir = getenv("H3_SUPER_INPUT_DIR");
    const char *output_dir = getenv("H3_SUPER_OUTPUT_DIR");
    const char *home = getenv("HOME");
    if (input_dir && *input_dir) {
        int length = snprintf(params->input_dir_storage,
                              sizeof(params->input_dir_storage), "%s",
                              input_dir);
        if (length > 0 && (size_t)length < sizeof(params->input_dir_storage)) {
            params->input_dir = params->input_dir_storage;
        }
    } else if (home && *home) {
        int length = snprintf(params->input_dir_storage,
                              sizeof(params->input_dir_storage),
                              "%s/ComfyUI-Shared/input", home);
        if (length > 0 && (size_t)length < sizeof(params->input_dir_storage)) {
            params->input_dir = params->input_dir_storage;
        }
    }
    if (output_dir && *output_dir) {
        int length = snprintf(params->output_dir_storage,
                              sizeof(params->output_dir_storage), "%s",
                              output_dir);
        if (length > 0 && (size_t)length < sizeof(params->output_dir_storage)) {
            params->output_dir = params->output_dir_storage;
        }
    } else if (home && *home) {
        int length = snprintf(params->output_dir_storage,
                              sizeof(params->output_dir_storage),
                              "%s/ComfyUI-Shared/output", home);
        if (length > 0 && (size_t)length < sizeof(params->output_dir_storage)) {
            params->output_dir = params->output_dir_storage;
        }
    }
    /* Preserve the official BF16 LoRA correction at runtime.  Requantizing
     * the merged base + delta to INT8 keeps the aggregate weight close but
     * can erase most of the much smaller LoRA correction.  The INT8 base
     * still provides the required 60--64 GiB memory/bandwidth envelope. */
    params->transformer = h3_super_official_transformer;
    params->refiner_lora = h3_super_official_refiner_lora;
    params->refiner_lora_strength = 0.8;
    params->text_encoder =
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";
    params->video_vae = "ltx-2.5-video-vae-bf16.safetensors";
    params->audio_vae = "ltx-2.5-audio-vae-bf16.safetensors";
    params->latent_upscaler =
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
    params->taehv_checkpoint = "taeltx2_3_wide.pth";
    params->taehv_compute_dtype = "bfloat16";
    params->taehv_temporal_mode = "auto";
    params->taehv_keep_on_device = 1;
    params->stage1_steps = 4;
    params->direct_handoff = 1;
    params->quality_attempts = 1;
    params->min_face_area = 0.0;
    params->negative_prompt =
        "cartoon, illustration, game graphics, low detail, blurry, text, watermark";
    params->timeout_seconds = 3600.0;
    params->poll_seconds = 1.0;
}

static int h3_super_validate_params(const h3_super_params *params,
                                    char *error, size_t error_size) {
    if (!params) return h3_super_fail(error, error_size,
                                      "Super parameters are NULL");
    if (params->profile != H3_SUPER_PROFILE_480P &&
        params->profile != H3_SUPER_PROFILE_480P_FAST &&
        params->profile != H3_SUPER_PROFILE_480P_QUALITY &&
        params->profile != H3_SUPER_PROFILE_480P_MOTION &&
        params->profile != H3_SUPER_PROFILE_V2) {
        return h3_super_fail(error, error_size,
                             "invalid Super profile value: %d",
                             (int)params->profile);
    }
    if (params->decoder != H3_SUPER_DECODER_TAEHV &&
        params->decoder != H3_SUPER_DECODER_OFFICIAL) {
        return h3_super_fail(error, error_size,
                             "invalid Super decoder value: %d",
                             (int)params->decoder);
    }
    if (params->stage2_schedule != H3_SUPER_STAGE2_SCHEDULE_DEFAULT &&
        params->stage2_schedule != H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9 &&
        params->stage2_schedule != H3_SUPER_STAGE2_SCHEDULE_TAIL_0P42) {
        return h3_super_fail(error, error_size,
                             "invalid Super Stage-2 schedule value: %d",
                             (int)params->stage2_schedule);
    }
    if (params->stage1_steps != 4 && params->stage1_steps != 6 &&
        params->stage1_steps != 8) {
        return h3_super_fail(error, error_size,
                             "Super Stage-1 steps must be 4, 6, or 8");
    }
    if (params->quality_attempts < 1 || params->quality_attempts > 16) {
        return h3_super_fail(error, error_size,
                             "Super quality attempts must be 1 through 16");
    }
    if (!isfinite(params->min_face_area) || params->min_face_area < 0.0 ||
        params->min_face_area > 0.25) {
        return h3_super_fail(error, error_size,
                             "Super minimum face area must be 0 through 0.25");
    }
    if (params->min_face_area > 0.0 &&
        !params->face_quality_gate && !params->person_quality_gate &&
        !params->output_quality_gate) {
        return h3_super_fail(error, error_size,
                             "Super minimum face area requires a Stage-1 or "
                             "refined-output quality gate");
    }
    if (params->quality_attempts > 1 &&
        !params->face_quality_gate && !params->person_quality_gate) {
        return h3_super_fail(error, error_size,
                             "multiple Super quality attempts require a "
                             "face or person quality gate");
    }
    if (params->quality_attempts > 1 &&
        (params->prefetch_transformer ||
         params->prefetch_conditioning || params->prefetch_input_models)) {
        return h3_super_fail(error, error_size,
                             "multiple Super quality attempts cannot prefetch "
                             "LTX before Stage-1 admission");
    }
    const int dimensions[] = {
        params->stage1_width, params->stage1_height, params->stage1_frames,
        params->input_width, params->input_height,
        params->refined_width, params->refined_height,
        params->output_width, params->output_height,
        params->output_frames, params->output_fps
    };
    for (size_t index = 0;
         index < sizeof(dimensions) / sizeof(dimensions[0]); index++) {
        if (dimensions[index] <= 0) {
            return h3_super_fail(error, error_size,
                                 "Super profile contains a non-positive dimension");
        }
    }
    if (params->stage1_frames < params->output_frames) {
        return h3_super_fail(error, error_size,
                             "Super Stage 1 has fewer frames than Stage 2");
    }
    if (params->refiner_lora &&
        (!*params->refiner_lora ||
         !isfinite(params->refiner_lora_strength) ||
         params->refiner_lora_strength <= 0.0)) {
        return h3_super_fail(error, error_size,
                             "Super refiner LoRA requires a positive finite "
                             "strength");
    }
    if (params->prefetch_transformer && params->prefetch_conditioning) {
        return h3_super_fail(error, error_size,
                             "Transformer and conditioning prefetch are "
                             "mutually exclusive");
    }
    if (params->prefetch_input_models && !params->prefetch_conditioning) {
        return h3_super_fail(error, error_size,
                             "input-model prefetch requires conditioning "
                             "prefetch so its load can be folded into the "
                             "conditioning prompt tail");
    }
    if (params->prepared_handoff && !params->direct_handoff) {
        return h3_super_fail(error, error_size,
                             "prepared BF16 handoff requires direct handoff");
    }
    if (params->fuse_denoise_decode &&
        params->decoder != H3_SUPER_DECODER_TAEHV) {
        return h3_super_fail(
            error, error_size,
            "fused denoise/decode currently requires the TAEHV decoder");
    }
    if (params->prefetch_start_step < 0 ||
        params->prefetch_start_step >= params->stage1_steps) {
        return h3_super_fail(error, error_size,
                             "Super prefetch start step must be 0 through %d",
                             params->stage1_steps - 1);
    }
    if (params->prefetch_start_step &&
        !params->prefetch_transformer &&
        !params->prefetch_conditioning) {
        return h3_super_fail(error, error_size,
                             "Super prefetch start step requires a prefetch mode");
    }
    if (params->prefetch_start_block < 0 ||
        params->prefetch_start_block > 50) {
        return h3_super_fail(error, error_size,
                             "Super prefetch start block must be 0 through 50");
    }
    if (params->final_evict_blocks < 0 ||
        params->final_evict_blocks > 50) {
        return h3_super_fail(error, error_size,
                             "Super final eviction group must be 0 through 50");
    }
    if (params->prefetch_start_block &&
        !params->prefetch_transformer &&
        !params->prefetch_conditioning) {
        return h3_super_fail(error, error_size,
                             "Super prefetch start block requires a prefetch mode");
    }
    if (params->prefetch_start_block && !params->final_evict_blocks) {
        return h3_super_fail(error, error_size,
                             "block-deadline prefetch requires final-pass eviction");
    }
    if (params->prefetch_start_block && params->prefetch_start_step) {
        return h3_super_fail(error, error_size,
                             "Super step and block prefetch deadlines are mutually exclusive");
    }
    if (params->refiner_lora &&
        (!h3_super_uses_runtime_refiner_base(params) ||
         strcmp(params->refiner_lora, h3_super_official_refiner_lora) ||
         fabs(params->refiner_lora_strength - 0.8) > 1e-12)) {
        return h3_super_fail(error, error_size,
                             "Super refiner LoRA is a pinned Sana contract: "
                             "use %s or %s + %s at strength 0.8",
                             h3_super_official_transformer,
                             h3_super_official_bf16_transformer,
                             h3_super_official_refiner_lora);
    }
    if (h3_super_uses_merged_refiner(params) && params->refiner_lora) {
        return h3_super_fail(error, error_size,
                             "the official merged Super refiner already "
                             "contains LoRA strength 0.8; do not also pass "
                             "--super-refiner-lora");
    }
    if (params->stage2_schedule != H3_SUPER_STAGE2_SCHEDULE_DEFAULT &&
        !h3_super_uses_merged_refiner(params) &&
        !(h3_super_uses_runtime_refiner_base(params) && params->refiner_lora &&
          !strcmp(params->refiner_lora, h3_super_official_refiner_lora) &&
          fabs(params->refiner_lora_strength - 0.8) <= 1e-12)) {
        return h3_super_fail(error, error_size,
                             "non-default Super Stage-2 schedules require "
                             "the provenance-verified official LTX refiner");
    }
    if ((params->input_width % 32) || (params->input_height % 32) ||
        params->refined_width != params->input_width * 2 ||
        params->refined_height != params->input_height * 2) {
        return h3_super_fail(error, error_size,
                             "Super LTX input must be 32-aligned and refined "
                             "dimensions must be exact learned x2 output");
    }
    if ((params->output_frames - 1) % 8) {
        return h3_super_fail(error, error_size,
                             "Super output frame count must be 8*n+1");
    }
    if (params->decoder == H3_SUPER_DECODER_TAEHV) {
        if (!params->taehv_checkpoint || !*params->taehv_checkpoint) {
            return h3_super_fail(error, error_size,
                                 "TAEHV checkpoint filename is empty");
        }
        if (!params->taehv_compute_dtype ||
            (strcmp(params->taehv_compute_dtype, "float16") &&
             strcmp(params->taehv_compute_dtype, "bfloat16"))) {
            return h3_super_fail(error, error_size,
                                 "TAEHV dtype must be float16 or bfloat16");
        }
        if (!params->taehv_temporal_mode ||
            (strcmp(params->taehv_temporal_mode, "auto") &&
             strcmp(params->taehv_temporal_mode, "parallel") &&
             strcmp(params->taehv_temporal_mode, "sequential"))) {
            return h3_super_fail(error, error_size,
                                 "TAEHV mode must be auto, parallel, or sequential");
        }
    }
    if (params->strict_person_quality) {
        if (params->profile != H3_SUPER_PROFILE_V2 ||
            params->stage1_steps != 4 ||
            params->stage2_schedule != H3_SUPER_STAGE2_SCHEDULE_DEFAULT) {
            return h3_super_fail(
                error, error_size,
                "person-strict-v1 requires profile v2, four H3 passes, "
                "and the default three-update LTX schedule");
        }
        if (!h3_super_uses_runtime_refiner_base(params) ||
            h3_super_uses_bf16_transformer(params) ||
            !params->refiner_lora ||
            strcmp(params->refiner_lora, h3_super_official_refiner_lora) ||
            fabs(params->refiner_lora_strength - 0.8) > 1e-12) {
            return h3_super_fail(
                error, error_size,
                "person-strict-v1 requires the official INT8 LTX base plus "
                "the official BF16 refiner LoRA runtime bypass at 0.8");
        }
        if (params->decoder != H3_SUPER_DECODER_TAEHV ||
            !params->taehv_temporal_mode ||
            strcmp(params->taehv_temporal_mode, "sequential") ||
            !params->taehv_compute_dtype ||
            strcmp(params->taehv_compute_dtype, "bfloat16") ||
            params->taehv_keep_on_device) {
            return h3_super_fail(
                error, error_size,
                "person-strict-v1 requires sequential BF16 TAEHV with "
                "post-decode eviction");
        }
        if (!params->sol_attention || !params->direct_handoff ||
            !params->prepared_handoff || !params->person_quality_gate ||
            !params->output_quality_gate || params->min_face_area < 0.006 ||
            params->quality_attempts < 2 || !params->allow_swap) {
            return h3_super_fail(
                error, error_size,
                "person-strict-v1 requires Sol-Attn, prepared BF16 handoff, "
                "both person gates, min face area 0.006, at least two "
                "candidate attempts, and explicit swap acceptance");
        }
        if (params->prefetch_transformer ||
            params->prefetch_conditioning ||
            params->prefetch_input_models || params->prefetch_start_step ||
            params->prefetch_start_block || params->final_evict_blocks ||
            params->fuse_denoise_decode) {
            return h3_super_fail(
                error, error_size,
                "person-strict-v1 disables cross-stage prefetch and fused "
                "decode so H3 admission completes before LTX residency");
        }
        if (!params->first_frame || !*params->first_frame) {
            return h3_super_fail(
                error, error_size,
                "person-strict-v1 requires a pinned --first-frame asset");
        }
    }
    if (params->first_frame) {
        struct stat info;
        if (!*params->first_frame ||
            stat(params->first_frame, &info) != 0 ||
            !S_ISREG(info.st_mode) || info.st_size <= 0 ||
            access(params->first_frame, R_OK) != 0) {
            return h3_super_fail(error, error_size,
                                 "Super first frame is missing or unreadable: %s",
                                 params->first_frame);
        }
    }
    return 1;
}

int h3_super_preflight(const h3_super_params *params,
                       h3_super_result *result,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!h3_super_validate_params(params, error, error_size)) return 0;
    if (params->strict_person_quality &&
        !h3_super_stage1_quality_contract(params, error, error_size)) {
        return 0;
    }
    double started = h3_super_now();
    if (!h3_super_directory(params->input_dir, 1, error, error_size) ||
        !h3_super_directory(params->output_dir, 1, error, error_size)) return 0;

    NSInteger status = 0;
    NSData *data = h3_super_http(params, @"GET", @"/object_info", nil,
                                 &status, error, error_size);
    if (!data) return 0;
    id object = h3_super_json_object(data, error, error_size);
    if (![object isKindOfClass:[NSDictionary class]]) {
        return h3_super_fail(error, error_size,
                             "ComfyUI /object_info is not a JSON object");
    }
    NSDictionary *classes = object;
    NSMutableArray<NSString *> *required = [NSMutableArray arrayWithArray:@[
        @"ImageScale", @"ImageFromBatch", @"VAEEncodeTiled",
        @"LTXVLatentUpsampler", @"LatentUpscaleModelLoader",
        @"LTXVAudioVAEEncode", @"LTXVConcatAVLatent",
        @"LTXVSeparateAVLatent", @"LTXVAddGuide", @"LTXVCropGuides",
        @"LTXVConditioning", @"BasicGuider",
        @"SamplerCustomAdvanced", @"ManualSigmas", @"KSamplerSelect",
        @"RandomNoise", @"CLIPTextEncode", @"CLIPLoader", @"UNETLoader",
        @"VAELoader", @"CreateVideo", @"SaveVideo",
        @"H3SuperSaveConditioning", @"H3SuperLoadConditioning",
        @"H3SuperSaveRefinerInput", @"H3SuperLoadRefinerInput",
        @"H3SuperSaveLatent", @"H3SuperLoadLatent"
    ]];
    if (params->direct_handoff) {
        [required addObject:@"H3SuperLoadHandoff"];
        [required addObject:@"H3SuperLoadHandoffAudio"];
    } else {
        [required addObject:@"LoadVideo"];
        [required addObject:@"VideoFrameSample"];
        [required addObject:@"GetVideoComponents"];
    }
    if (params->first_frame) {
        [required addObject:@"H3SuperLoadOpeningFrame"];
        [required addObject:@"H3SuperPrepareOfficialLTX25ImageGuide"];
    }
    if (params->decoder == H3_SUPER_DECODER_TAEHV) {
        [required addObject:@"H3SuperTAEHVDecode"];
    } else {
        [required addObject:@"VAEDecodeTiled"];
    }
    if (params->prefetch_transformer) {
        [required addObject:@"H3SuperPrefetchModel"];
        [required addObject:@"H3SuperLoadPrefetchedModel"];
    }
    if (params->prefetch_input_models) {
        [required addObject:@"H3SuperLoadPrefetchedInputModels"];
    }
    if (params->refiner_lora) {
        [required addObject:@"H3SuperApplyOfficialRefinerLoRA"];
    }
    if (h3_super_uses_merged_refiner(params)) {
        [required addObject:@"H3SuperVerifyOfficialMergedRefiner"];
    }
    if (params->sol_attention) {
        [required addObject:@"H3SuperApplySolAttention"];
    }
    for (NSString *name in required) {
        if (![classes objectForKey:name]) {
            return h3_super_fail(error, error_size,
                                 "ComfyUI is missing required node: %s",
                                 name.UTF8String);
        }
    }
    const char *models[] = {
        params->transformer, params->text_encoder, params->video_vae,
        params->audio_vae, params->latent_upscaler
    };
    for (size_t index = 0; index < sizeof(models) / sizeof(models[0]); index++) {
        NSString *name = h3_super_string(models[index]);
        if (!name.length || !h3_super_object_contains(classes, name)) {
            return h3_super_fail(error, error_size,
                                 "ComfyUI does not advertise configured model: %s",
                                 models[index] ? models[index] : "(null)");
        }
    }
    if (params->refiner_lora) {
        NSString *name = h3_super_string(params->refiner_lora);
        if (!name.length || !h3_super_object_contains(classes, name)) {
            return h3_super_fail(error, error_size,
                                 "ComfyUI does not advertise configured refiner "
                                 "LoRA: %s", params->refiner_lora);
        }
    }
    if (params->decoder == H3_SUPER_DECODER_TAEHV) {
        NSString *name = h3_super_string(params->taehv_checkpoint);
        if (!name.length || !h3_super_object_contains(classes, name)) {
            return h3_super_fail(error, error_size,
                                 "ComfyUI does not advertise configured TAEHV "
                                 "checkpoint: %s",
                                 params->taehv_checkpoint ?
                                 params->taehv_checkpoint : "(null)");
        }
    }
    if (!h3_super_release_backend(params, error, error_size)) return 0;
    if (result) {
        result->swapouts_available = h3_super_swapouts(
            &result->system_swapouts_before);
        result->no_swap_gate_passed = 0;
        h3_super_system_memory(params, &result->backend_ram_total,
                               &result->backend_ram_free_before);
        result->backend_ram_free_min = result->backend_ram_free_before;
    }
    if (result) result->preflight_seconds = h3_super_now() - started;
    return 1;
}

int h3_super_make_stage1_path(const h3_super_params *params,
                              char *path, size_t path_size,
                              char *error, size_t error_size) {
    if (!params || !path || path_size == 0) {
        return h3_super_fail(error, error_size,
                             "invalid Stage-1 path arguments");
    }
    NSString *directory = h3_super_string(params->input_dir);
    NSString *extension = params->direct_handoff ? @"h3sh" : @"mp4";
    NSString *name = [NSString stringWithFormat:@"h3-super-%d-%@.%@",
                      getpid(), NSUUID.UUID.UUIDString.lowercaseString,
                      extension];
    NSString *candidate = [directory stringByAppendingPathComponent:name];
    if (!h3_super_copy_string(path, path_size, candidate, error, error_size)) {
        return 0;
    }
    if (access(path, F_OK) == 0) {
        return h3_super_fail(error, error_size,
                             "unique Stage-1 path already exists: %s", path);
    }
    return 1;
}

int h3_super_import_stage1(const char *source_path,
                           const char *stage1_path,
                           char *error, size_t error_size) {
    if (!source_path || !*source_path || !stage1_path || !*stage1_path) {
        return h3_super_fail(error, error_size,
                             "invalid Stage-1 import arguments");
    }
    struct stat source_info;
    if (stat(source_path, &source_info) != 0 ||
        !S_ISREG(source_info.st_mode) || source_info.st_size <= 0) {
        return h3_super_fail(error, error_size,
                             "source Stage-1 artifact is missing or empty: %s",
                             source_path);
    }
    NSError *file_error = nil;
    if (![NSFileManager.defaultManager
            copyItemAtPath:h3_super_string(source_path)
                    toPath:h3_super_string(stage1_path)
                     error:&file_error]) {
        return h3_super_fail(error, error_size,
                             "cannot import Stage-1 artifact: %s",
                             file_error.localizedDescription.UTF8String);
    }
    return 1;
}

static int h3_super_write_all(int descriptor, const void *data, size_t bytes,
                              char *error, size_t error_size) {
    const uint8_t *cursor = data;
    while (bytes) {
        ssize_t written = write(descriptor, cursor, bytes);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            return h3_super_fail(error, error_size,
                                 "cannot write direct handoff: %s",
                                 strerror(errno));
        }
        cursor += (size_t)written;
        bytes -= (size_t)written;
    }
    return 1;
}

static int h3_super_finite_range(const float *values, size_t count,
                                 float minimum, float maximum) {
    if (!values) return 0;
    for (size_t index = 0; index < count; index++) {
        float value = values[index];
        if (!isfinite(value) || value < minimum || value > maximum) return 0;
    }
    return 1;
}

static uint16_t h3_super_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & UINT32_C(1));
    return (uint16_t)(bits >> 16);
}

static int h3_super_resize_axis_build(int input_size, int output_size,
                                      int crop_offset,
                                      h3_super_resize_axis *axis) {
    if (input_size < 1 || output_size < 1 || crop_offset < 0 || !axis)
        return 0;
    double scale = (double)input_size / (double)output_size;
    for (int output = 0; output < output_size; output++) {
        double source = ((double)output + 0.5) * scale - 0.5;
        if (source < 0.0) source = 0.0;
        if (source > (double)(input_size - 1))
            source = (double)(input_size - 1);
        int low = (int)floor(source);
        int high = low < input_size - 1 ? low + 1 : low;
        axis[output].low = crop_offset + low;
        axis[output].high = crop_offset + high;
        axis[output].high_weight = (float)(source - (double)low);
    }
    return 1;
}

static int h3_super_prepare_video_bf16(const h3_super_params *params,
                                       const h3_result *stage1,
                                       uint16_t **prepared,
                                       size_t *prepared_elements,
                                       char *error, size_t error_size) {
    if (prepared) *prepared = NULL;
    if (prepared_elements) *prepared_elements = 0;
    if (!params || !stage1 || !prepared || !prepared_elements ||
        !stage1->decoded_rgb || params->output_frames < 1 ||
        params->input_width < 1 || params->input_height < 1) {
        return h3_super_fail(error, error_size,
                             "invalid direct handoff video preparation");
    }
    int source_width = stage1->decoded_width;
    int source_height = stage1->decoded_height;
    int target_width = params->input_width;
    int target_height = params->input_height;
    double source_aspect = (double)source_width / (double)source_height;
    double target_aspect = (double)target_width / (double)target_height;
    int crop_x = 0;
    int crop_y = 0;
    if (source_aspect > target_aspect) {
        crop_x = (int)nearbyint(
            ((double)source_width -
             (double)source_width * (target_aspect / source_aspect)) /
            2.0);
    } else if (source_aspect < target_aspect) {
        crop_y = (int)nearbyint(
            ((double)source_height -
             (double)source_height * (source_aspect / target_aspect)) /
            2.0);
    }
    int cropped_width = source_width - 2 * crop_x;
    int cropped_height = source_height - 2 * crop_y;
    if (cropped_width < 1 || cropped_height < 1) {
        return h3_super_fail(error, error_size,
                             "direct handoff center crop is empty");
    }

    size_t target_area = (size_t)target_width * (size_t)target_height;
    if (target_area > SIZE_MAX / 3 ||
        (size_t)params->output_frames > SIZE_MAX / (target_area * 3)) {
        return h3_super_fail(error, error_size,
                             "direct handoff BF16 video size overflows");
    }
    size_t elements = (size_t)params->output_frames * target_area * 3;
    if (elements > SIZE_MAX / sizeof(uint16_t)) {
        return h3_super_fail(error, error_size,
                             "direct handoff BF16 payload size overflows");
    }
    h3_super_resize_axis *x_axis = malloc(
        (size_t)target_width * sizeof(*x_axis));
    h3_super_resize_axis *y_axis = malloc(
        (size_t)target_height * sizeof(*y_axis));
    uint16_t *video = malloc(elements * sizeof(*video));
    if (!x_axis || !y_axis || !video ||
        !h3_super_resize_axis_build(cropped_width, target_width,
                                    crop_x, x_axis) ||
        !h3_super_resize_axis_build(cropped_height, target_height,
                                    crop_y, y_axis)) {
        free(x_axis);
        free(y_axis);
        free(video);
        return h3_super_fail(error, error_size,
                             "cannot allocate direct BF16 handoff video");
    }

    size_t source_area = (size_t)source_width * (size_t)source_height;
    size_t source_frame_elements = source_area * 3;
    size_t target_frame_elements = target_area * 3;
    for (int frame = 0; frame < params->output_frames; frame++) {
        const float *source = stage1->decoded_rgb +
            (size_t)frame * source_frame_elements;
        uint16_t *target = video + (size_t)frame * target_frame_elements;
        for (int y = 0; y < target_height; y++) {
            int y0 = y_axis[y].low;
            int y1 = y_axis[y].high;
            float wy = y_axis[y].high_weight;
            float wy0 = 1.0f - wy;
            for (int x = 0; x < target_width; x++) {
                int x0 = x_axis[x].low;
                int x1 = x_axis[x].high;
                float wx = x_axis[x].high_weight;
                float wx0 = 1.0f - wx;
                size_t top_left =
                    ((size_t)y0 * (size_t)source_width + (size_t)x0) * 3;
                size_t top_right =
                    ((size_t)y0 * (size_t)source_width + (size_t)x1) * 3;
                size_t bottom_left =
                    ((size_t)y1 * (size_t)source_width + (size_t)x0) * 3;
                size_t bottom_right =
                    ((size_t)y1 * (size_t)source_width + (size_t)x1) * 3;
                size_t destination =
                    ((size_t)y * (size_t)target_width + (size_t)x) * 3;
                for (int channel = 0; channel < 3; channel++) {
                    float top = source[top_left + (size_t)channel] * wx0 +
                                source[top_right + (size_t)channel] * wx;
                    float bottom =
                        source[bottom_left + (size_t)channel] * wx0 +
                        source[bottom_right + (size_t)channel] * wx;
                    float value = top * wy0 + bottom * wy;
                    if (!isfinite(value)) {
                        free(x_axis);
                        free(y_axis);
                        free(video);
                        return h3_super_fail(
                            error, error_size,
                            "Stage-1 video contains invalid F32 pixels");
                    }
                    if (value < 0.0f) value = 0.0f;
                    if (value > 1.0f) value = 1.0f;
                    target[destination + (size_t)channel] =
                        h3_super_f32_to_bf16(value);
                }
            }
        }
    }
    free(x_axis);
    free(y_axis);
    *prepared = video;
    *prepared_elements = elements;
    return 1;
}

int h3_super_write_handoff(const h3_super_params *params,
                           const h3_result *stage1,
                           const char *handoff_path,
                           uint64_t *bytes,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (bytes) *bytes = 0;
    if (!params || !stage1 || !handoff_path || !*handoff_path ||
        !params->direct_handoff) {
        return h3_super_fail(error, error_size,
                             "invalid direct handoff arguments");
    }
    if (!stage1->decoded_rgb || !stage1->audio_pcm ||
        stage1->decoded_width != params->stage1_width ||
        stage1->decoded_height != params->stage1_height ||
        stage1->decoded_frames < params->output_frames ||
        stage1->audio_channels != 2 || stage1->sample_rate != 32000 ||
        params->output_fps < 1) {
        return h3_super_fail(error, error_size,
                             "retained Stage-1 tensors violate the direct "
                             "handoff contract");
    }
    int64_t target_samples_64 =
        (int64_t)params->output_frames * stage1->sample_rate /
        params->output_fps;
    if (target_samples_64 < 1 || target_samples_64 > stage1->audio_samples ||
        target_samples_64 > INT32_MAX) {
        return h3_super_fail(error, error_size,
                             "Stage-1 PCM is shorter than the Super output");
    }
    int target_samples = (int)target_samples_64;
    size_t source_area = (size_t)stage1->decoded_width *
                         (size_t)stage1->decoded_height;
    if (source_area > SIZE_MAX / 3 ||
        (size_t)params->output_frames > SIZE_MAX / (source_area * 3)) {
        return h3_super_fail(error, error_size,
                             "direct handoff source video size overflows");
    }
    size_t source_video_elements =
        (size_t)params->output_frames * source_area * 3;
    if (!h3_super_finite_range(stage1->decoded_rgb, source_video_elements,
                               -0.0001f, 1.0001f)) {
        return h3_super_fail(error, error_size,
                             "Stage-1 video contains invalid F32 pixels");
    }
    uint16_t *prepared_video = NULL;
    const void *video_payload = stage1->decoded_rgb;
    size_t video_elements = source_video_elements;
    if (params->prepared_handoff) {
        if (!h3_super_prepare_video_bf16(
                params, stage1, &prepared_video, &video_elements,
                error, error_size)) return 0;
        video_payload = prepared_video;
    }
    size_t audio_elements = (size_t)stage1->audio_channels *
                            (size_t)target_samples;
    if (audio_elements > SIZE_MAX / sizeof(float)) {
        free(prepared_video);
        return h3_super_fail(error, error_size,
                             "direct handoff payload size overflows");
    }
    size_t video_item_bytes = params->prepared_handoff ?
        sizeof(*prepared_video) : sizeof(float);
    if (video_elements > SIZE_MAX / video_item_bytes) {
        free(prepared_video);
        return h3_super_fail(error, error_size,
                             "direct handoff video payload size overflows");
    }
    size_t video_bytes = video_elements * video_item_bytes;
    size_t audio_bytes = audio_elements * sizeof(float);
    uint64_t video_offset = H3_SUPER_HANDOFF_HEADER_BYTES;
    uint64_t audio_offset = (video_offset + video_bytes + UINT64_C(63)) &
                            ~UINT64_C(63);
    uint64_t file_bytes = audio_offset + audio_bytes;
    if (file_bytes > (uint64_t)INT64_MAX) {
        free(prepared_video);
        return h3_super_fail(error, error_size,
                             "direct handoff file size overflows");
    }
    for (int channel = 0; channel < stage1->audio_channels; channel++) {
        const float *channel_pcm = stage1->audio_pcm +
            (size_t)channel * (size_t)stage1->audio_samples;
        if (!h3_super_finite_range(channel_pcm, (size_t)target_samples,
                                   -1.0001f, 1.0001f)) {
            free(prepared_video);
            return h3_super_fail(error, error_size,
                                 "Stage-1 audio contains invalid F32 PCM");
        }
    }

    char header[H3_SUPER_HANDOFF_HEADER_BYTES];
    memset(header, 0, sizeof(header));
    int header_length;
    if (params->prepared_handoff) {
        header_length = snprintf(
            header, sizeof(header),
            "%s{\"schema\":\"h3-super-handoff-v2\","
            "\"header_bytes\":%d,\"file_bytes\":%" PRIu64 ","
            "\"video\":{\"dtype\":\"bfloat16\",\"layout\":\"THWC\","
            "\"frames\":%d,\"height\":%d,\"width\":%d,\"channels\":3,"
            "\"fps\":%d,\"offset\":%" PRIu64 ",\"bytes\":%zu,"
            "\"range\":\"zero_one\",\"preprocess\":{"
            "\"source_height\":%d,\"source_width\":%d,"
            "\"crop\":\"center\",\"resize\":\"bilinear\","
            "\"align_corners\":false}},"
            "\"audio\":{\"dtype\":\"float32\",\"layout\":\"CS\","
            "\"channels\":%d,\"samples\":%d,\"sample_rate\":%d,"
            "\"offset\":%" PRIu64 ",\"bytes\":%zu,"
            "\"range\":\"minus_one_one\"}}\n",
            h3_super_handoff_magic, H3_SUPER_HANDOFF_HEADER_BYTES, file_bytes,
            params->output_frames, params->input_height, params->input_width,
            params->output_fps, video_offset, video_bytes,
            stage1->decoded_height, stage1->decoded_width,
            stage1->audio_channels, target_samples, stage1->sample_rate,
            audio_offset, audio_bytes);
    } else {
        header_length = snprintf(
            header, sizeof(header),
            "%s{\"schema\":\"h3-super-handoff-v1\","
            "\"header_bytes\":%d,\"file_bytes\":%" PRIu64 ","
            "\"video\":{\"dtype\":\"float32\",\"layout\":\"THWC\","
            "\"frames\":%d,\"height\":%d,\"width\":%d,\"channels\":3,"
            "\"fps\":%d,\"offset\":%" PRIu64 ",\"bytes\":%zu,"
            "\"range\":\"zero_one\"},"
            "\"audio\":{\"dtype\":\"float32\",\"layout\":\"CS\","
            "\"channels\":%d,\"samples\":%d,\"sample_rate\":%d,"
            "\"offset\":%" PRIu64 ",\"bytes\":%zu,"
            "\"range\":\"minus_one_one\"}}\n",
            h3_super_handoff_magic, H3_SUPER_HANDOFF_HEADER_BYTES, file_bytes,
            params->output_frames, stage1->decoded_height,
            stage1->decoded_width, params->output_fps,
            video_offset, video_bytes, stage1->audio_channels,
            target_samples, stage1->sample_rate, audio_offset, audio_bytes);
    }
    if (header_length < 0 || (size_t)header_length >= sizeof(header)) {
        free(prepared_video);
        return h3_super_fail(error, error_size,
                             "direct handoff header is too large");
    }
    char temporary[H3_SUPER_PATH_MAX];
    int temporary_length = snprintf(temporary, sizeof(temporary),
                                    "%s.partial-%d", handoff_path, getpid());
    if (temporary_length < 0 || (size_t)temporary_length >= sizeof(temporary)) {
        free(prepared_video);
        return h3_super_fail(error, error_size,
                             "direct handoff path is too long");
    }
    int descriptor = open(temporary, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (descriptor < 0) {
        free(prepared_video);
        return h3_super_fail(error, error_size,
                             "cannot create direct handoff: %s",
                             strerror(errno));
    }
    int ok = h3_super_write_all(descriptor, header, sizeof(header),
                                error, error_size) &&
             h3_super_write_all(descriptor, video_payload, video_bytes,
                                error, error_size);
    free(prepared_video);
    uint64_t position = video_offset + video_bytes;
    if (ok && position < audio_offset) {
        uint8_t padding[64] = {0};
        ok = h3_super_write_all(descriptor, padding,
                                (size_t)(audio_offset - position),
                                error, error_size);
    }
    for (int channel = 0; ok && channel < stage1->audio_channels; channel++) {
        const float *channel_pcm = stage1->audio_pcm +
            (size_t)channel * (size_t)stage1->audio_samples;
        ok = h3_super_write_all(descriptor, channel_pcm,
                                (size_t)target_samples * sizeof(float),
                                error, error_size);
    }
    if (close(descriptor) != 0 && ok) {
        ok = h3_super_fail(error, error_size,
                           "cannot close direct handoff: %s", strerror(errno));
    }
    if (ok && rename(temporary, handoff_path) != 0) {
        ok = h3_super_fail(error, error_size,
                           "cannot publish direct handoff: %s", strerror(errno));
    }
    if (!ok) {
        unlink(temporary);
        return 0;
    }
    if (bytes) *bytes = file_bytes;
    return 1;
}

static NSArray *h3_super_link(NSString *node, NSInteger output) {
    return @[node, @(output)];
}

static NSDictionary *h3_super_node(NSString *type, NSDictionary *inputs) {
    return @{ @"class_type": type, @"inputs": inputs };
}

static NSDictionary *h3_super_conditioning_workflow(
        const h3_super_params *params, NSString *prompt,
        NSString *state_name) {
    NSString *negative = h3_super_string(params->negative_prompt);
    if (!negative) negative = @"";
    NSMutableDictionary *nodes = [NSMutableDictionary dictionary];
    nodes[@"clip"] = h3_super_node(@"CLIPLoader", @{
        @"clip_name": h3_super_string(params->text_encoder),
        @"type": @"ltxv", @"device": @"default"
    });
    nodes[@"positive"] = h3_super_node(@"CLIPTextEncode", @{
        @"text": prompt, @"clip": h3_super_link(@"clip", 0)
    });
    nodes[@"negative"] = h3_super_node(@"CLIPTextEncode", @{
        @"text": negative, @"clip": h3_super_link(@"clip", 0)
    });
    nodes[@"conditioning"] = h3_super_node(@"LTXVConditioning", @{
        @"positive": h3_super_link(@"positive", 0),
        @"negative": h3_super_link(@"negative", 0),
        @"frame_rate": @(params->output_fps)
    });
    NSMutableDictionary *save_inputs = [@{
        @"positive": h3_super_link(@"conditioning", 0),
        @"negative": h3_super_link(@"conditioning", 1),
        @"filename": state_name
    } mutableCopy];
    if (params->prefetch_input_models) {
        nodes[@"video_vae"] = h3_super_node(@"VAELoader", @{
            @"vae_name": h3_super_string(params->video_vae)
        });
        nodes[@"audio_vae"] = h3_super_node(@"VAELoader", @{
            @"vae_name": h3_super_string(params->audio_vae)
        });
        nodes[@"upscaler"] = h3_super_node(@"LatentUpscaleModelLoader", @{
            @"model_name": h3_super_string(params->latent_upscaler)
        });
        save_inputs[@"video_vae"] = h3_super_link(@"video_vae", 0);
        save_inputs[@"audio_vae"] = h3_super_link(@"audio_vae", 0);
        save_inputs[@"upscale_model"] = h3_super_link(@"upscaler", 0);
    }
    nodes[@"save_state"] = h3_super_node(
        @"H3SuperSaveConditioning", save_inputs);
    return nodes;
}

static NSDictionary *h3_super_transformer_prefetch_workflow(
        const h3_super_params *params) {
    NSMutableDictionary *nodes = [NSMutableDictionary dictionary];
    nodes[@"transformer"] = h3_super_node(@"UNETLoader", @{
        @"unet_name": h3_super_string(params->transformer),
        @"weight_dtype": @"default"
    });
    NSString *model_node = @"transformer";
    if (h3_super_uses_merged_refiner(params)) {
        nodes[@"merged_refiner"] = h3_super_node(
            @"H3SuperVerifyOfficialMergedRefiner", @{
                @"model": h3_super_link(@"transformer", 0),
                @"transformer_name": h3_super_string(params->transformer)
            });
        model_node = @"merged_refiner";
    } else if (params->refiner_lora) {
        nodes[@"refiner_lora"] = h3_super_node(
            @"H3SuperApplyOfficialRefinerLoRA", @{
                @"model": h3_super_link(@"transformer", 0),
                @"transformer_name": h3_super_string(params->transformer),
                @"lora_name": h3_super_string(params->refiner_lora),
                @"strength_model": @(params->refiner_lora_strength)
            });
        model_node = @"refiner_lora";
    }
    nodes[@"prefetch"] = h3_super_node(@"H3SuperPrefetchModel", @{
        @"model": h3_super_link(model_node, 0)
    });
    return nodes;
}

static NSDictionary *h3_super_input_models_prefetch_workflow(
        const h3_super_params *params) {
    NSMutableDictionary *nodes = [NSMutableDictionary dictionary];
    nodes[@"video_vae"] = h3_super_node(@"VAELoader", @{
        @"vae_name": h3_super_string(params->video_vae)
    });
    nodes[@"audio_vae"] = h3_super_node(@"VAELoader", @{
        @"vae_name": h3_super_string(params->audio_vae)
    });
    nodes[@"upscaler"] = h3_super_node(@"LatentUpscaleModelLoader", @{
        @"model_name": h3_super_string(params->latent_upscaler)
    });
    nodes[@"prefetch"] = h3_super_node(@"H3SuperPrefetchInputModels", @{
        @"video_vae": h3_super_link(@"video_vae", 0),
        @"audio_vae": h3_super_link(@"audio_vae", 0),
        @"upscale_model": h3_super_link(@"upscaler", 0)
    });
    return nodes;
}

static NSDictionary *h3_super_input_workflow(
        const h3_super_params *params, NSString *input_name,
        NSString *conditioning_name, NSString *state_name,
        NSString *opening_frame_name, NSString *opening_frame_sha256,
        BOOL input_models_prefetched) {
    NSMutableDictionary *nodes = [NSMutableDictionary dictionary];
    NSArray *source_pixels = nil;
    NSArray *source_audio = nil;
    if (params->direct_handoff) {
        nodes[@"handoff"] = h3_super_node(@"H3SuperLoadHandoff", @{
            @"filename": input_name,
            @"expected_frames": @(params->output_frames),
            @"expected_width": @(params->input_width),
            @"expected_height": @(params->input_height),
            @"expected_source_width": @(params->stage1_width),
            @"expected_source_height": @(params->stage1_height)
        });
        source_pixels = h3_super_link(@"handoff", 0);
        source_audio = h3_super_link(@"handoff", 1);
    } else {
        nodes[@"load_video"] = h3_super_node(@"LoadVideo", @{
            @"file": input_name
        });
        nodes[@"trim"] = h3_super_node(@"VideoFrameSample", @{
            @"video": h3_super_link(@"load_video", 0),
            @"num_frames": @(params->output_frames),
            @"strategy": @"head", @"seed": @0
        });
        nodes[@"components"] = h3_super_node(@"GetVideoComponents", @{
            @"video": h3_super_link(@"trim", 0)
        });
        source_pixels = h3_super_link(@"components", 0);
        source_audio = h3_super_link(@"components", 1);
    }
    NSArray *low_pixels = source_pixels;
    if (!params->direct_handoff) {
        nodes[@"low_pixels"] = h3_super_node(@"ImageScale", @{
            @"image": source_pixels,
            @"upscale_method": @"lanczos",
            @"width": @(params->input_width),
            @"height": @(params->input_height), @"crop": @"center"
        });
        low_pixels = h3_super_link(@"low_pixels", 0);
    }
    NSArray *video_vae = nil;
    NSArray *audio_vae = nil;
    NSArray *upscale_model = nil;
    if (input_models_prefetched) {
        nodes[@"input_models"] = h3_super_node(
            @"H3SuperLoadPrefetchedInputModels", @{});
        video_vae = h3_super_link(@"input_models", 0);
        audio_vae = h3_super_link(@"input_models", 1);
        upscale_model = h3_super_link(@"input_models", 2);
    } else {
        nodes[@"video_vae"] = h3_super_node(@"VAELoader", @{
            @"vae_name": h3_super_string(params->video_vae)
        });
        nodes[@"audio_vae"] = h3_super_node(@"VAELoader", @{
            @"vae_name": h3_super_string(params->audio_vae)
        });
        nodes[@"upscaler_loader"] = h3_super_node(
            @"LatentUpscaleModelLoader", @{
                @"model_name": h3_super_string(params->latent_upscaler)
            });
        video_vae = h3_super_link(@"video_vae", 0);
        audio_vae = h3_super_link(@"audio_vae", 0);
        upscale_model = h3_super_link(@"upscaler_loader", 0);
    }
    nodes[@"encoded"] = h3_super_node(@"VAEEncodeTiled", @{
        @"pixels": low_pixels,
        @"vae": video_vae,
        @"tile_size": @768, @"overlap": @64,
        @"temporal_size": @128, @"temporal_overlap": @24
    });
    nodes[@"upsampled"] = h3_super_node(@"LTXVLatentUpsampler", @{
        @"samples": h3_super_link(@"encoded", 0),
        @"upscale_model": upscale_model,
        @"vae": video_vae
    });
    NSArray *guide_source = nil;
    if (opening_frame_name.length) {
        nodes[@"opening_frame"] = h3_super_node(
            @"H3SuperLoadOpeningFrame", @{
                @"filename": opening_frame_name,
                @"expected_sha256": opening_frame_sha256
            });
        nodes[@"official_guide"] = h3_super_node(
            @"H3SuperPrepareOfficialLTX25ImageGuide", @{
                @"image": h3_super_link(@"opening_frame", 0),
                @"crf": @18
            });
        guide_source = h3_super_link(@"official_guide", 0);
    } else {
        nodes[@"first_frame"] = h3_super_node(@"ImageFromBatch", @{
            @"image": source_pixels, @"batch_index": @0,
            @"length": @1
        });
        guide_source = h3_super_link(@"first_frame", 0);
    }
    nodes[@"conditioning"] = h3_super_node(@"H3SuperLoadConditioning", @{
        @"filename": conditioning_name
    });
    nodes[@"guide"] = h3_super_node(@"LTXVAddGuide", @{
        @"positive": h3_super_link(@"conditioning", 0),
        @"negative": h3_super_link(@"conditioning", 1),
        @"vae": video_vae,
        @"latent": h3_super_link(@"upsampled", 0),
        @"image": guide_source,
        @"frame_idx": @0, @"strength": @1.0
    });
    nodes[@"audio_encoded"] = h3_super_node(@"LTXVAudioVAEEncode", @{
        @"audio": source_audio,
        @"audio_vae": audio_vae
    });
    nodes[@"joint"] = h3_super_node(@"LTXVConcatAVLatent", @{
        @"video_latent": h3_super_link(@"guide", 2),
        @"audio_latent": h3_super_link(@"audio_encoded", 0)
    });
    nodes[@"save_state"] = h3_super_node(@"H3SuperSaveRefinerInput", @{
        @"positive": h3_super_link(@"guide", 0),
        @"negative": h3_super_link(@"guide", 1),
        @"latent": h3_super_link(@"joint", 0),
        @"filename": state_name
    });
    return nodes;
}

static int h3_super_prepare_first_frame(
        const h3_super_params *params, NSString *identifier,
        h3_super_result *result, NSString **relative_path,
        NSString **absolute_path, char *error, size_t error_size) {
    if (relative_path) *relative_path = nil;
    if (absolute_path) *absolute_path = nil;
    if (!params->first_frame) return 1;

    double started = h3_super_now();
    NSString *source = [h3_super_string(params->first_frame)
        stringByStandardizingPath];
    NSString *extension = source.pathExtension.lowercaseString;
    NSCharacterSet *unsafe = [NSCharacterSet
        characterSetWithCharactersInString:
            @"abcdefghijklmnopqrstuvwxyz0123456789"].invertedSet;
    if (!extension.length || extension.length > 16 ||
        [extension rangeOfCharacterFromSet:unsafe].location != NSNotFound) {
        extension = @"img";
    }
    NSString *name = [NSString stringWithFormat:
        @"h3-super-%@-opening.%@", identifier, extension];
    NSString *destination = [h3_super_string(params->input_dir)
        stringByAppendingPathComponent:name];
    NSString *temporary = [destination stringByAppendingFormat:
        @".partial-%d", getpid()];
    NSFileManager *manager = NSFileManager.defaultManager;
    [manager removeItemAtPath:temporary error:nil];
    [manager removeItemAtPath:destination error:nil];
    NSError *file_error = nil;
    if (![manager copyItemAtPath:source toPath:temporary error:&file_error]) {
        [manager removeItemAtPath:temporary error:nil];
        return h3_super_fail(error, error_size,
                             "cannot import Super first frame: %s",
                             file_error.localizedDescription.UTF8String);
    }
    if (![manager moveItemAtPath:temporary toPath:destination
                            error:&file_error]) {
        [manager removeItemAtPath:temporary error:nil];
        return h3_super_fail(error, error_size,
                             "cannot install Super first frame: %s",
                             file_error.localizedDescription.UTF8String);
    }
    uint64_t source_bytes = 0;
    int ok = h3_super_sha256_file(
        source.fileSystemRepresentation, result->first_frame_sha256,
        &source_bytes, error, error_size) &&
        h3_super_sha256_file(
            destination.fileSystemRepresentation,
            result->first_frame_copy_sha256, NULL, error, error_size);
    if (!ok || strcmp(result->first_frame_sha256,
                      result->first_frame_copy_sha256)) {
        [manager removeItemAtPath:destination error:nil];
        if (ok) {
            return h3_super_fail(error, error_size,
                                 "Super first-frame SHA-256 changed during import");
        }
        return 0;
    }
    result->first_frame_bytes = source_bytes;
    result->first_frame_import_seconds = h3_super_now() - started;
    result->original_first_frame_conditioning = 1;
    if (relative_path) *relative_path = name;
    if (absolute_path) *absolute_path = destination;
    return 1;
}

static NSDictionary *h3_super_denoise_workflow(
        const h3_super_params *params, NSString *input_state_name,
        NSString *output_state_name, uint64_t seed) {
    NSMutableDictionary *nodes = [NSMutableDictionary dictionary];
    nodes[@"state"] = h3_super_node(@"H3SuperLoadRefinerInput", @{
        @"filename": input_state_name
    });
    if (params->prefetch_transformer) {
        nodes[@"transformer"] = h3_super_node(
            @"H3SuperLoadPrefetchedModel", @{});
    } else {
        nodes[@"transformer"] = h3_super_node(@"UNETLoader", @{
            @"unet_name": h3_super_string(params->transformer),
            @"weight_dtype": @"default"
        });
    }
    NSString *denoiser_model = @"transformer";
    if (h3_super_uses_merged_refiner(params) &&
        !params->prefetch_transformer) {
        nodes[@"merged_refiner"] = h3_super_node(
            @"H3SuperVerifyOfficialMergedRefiner", @{
                @"model": h3_super_link(@"transformer", 0),
                @"transformer_name": h3_super_string(params->transformer)
            });
        denoiser_model = @"merged_refiner";
    } else if (params->refiner_lora && !params->prefetch_transformer) {
        nodes[@"refiner_lora"] = h3_super_node(
            @"H3SuperApplyOfficialRefinerLoRA", @{
                @"model": h3_super_link(@"transformer", 0),
                @"transformer_name": h3_super_string(params->transformer),
                @"lora_name": h3_super_string(params->refiner_lora),
                @"strength_model": @(params->refiner_lora_strength)
            });
        denoiser_model = @"refiner_lora";
    }
    if (params->sol_attention) {
        nodes[@"sol_attention"] = h3_super_node(
            @"H3SuperApplySolAttention", @{
                @"model": h3_super_link(denoiser_model, 0),
                @"query_block_size": @"64",
                @"schedule": h3_super_string(
                    h3_super_stage2_schedule_name(params->stage2_schedule))
            });
        denoiser_model = @"sol_attention";
    }
    nodes[@"guider"] = h3_super_node(@"BasicGuider", @{
        @"model": h3_super_link(denoiser_model, 0),
        @"conditioning": h3_super_link(@"state", 0)
    });
    nodes[@"noise"] = h3_super_node(@"RandomNoise", @{
        @"noise_seed": @(seed)
    });
    nodes[@"sampler"] = h3_super_node(@"KSamplerSelect", @{
        @"sampler_name": @"euler"
    });
    nodes[@"sigmas"] = h3_super_node(@"ManualSigmas", @{
        @"sigmas": h3_super_string(
            h3_super_stage2_sigmas(params->stage2_schedule))
    });
    nodes[@"denoise"] = h3_super_node(@"SamplerCustomAdvanced", @{
        @"noise": h3_super_link(@"noise", 0),
        @"guider": h3_super_link(@"guider", 0),
        @"sampler": h3_super_link(@"sampler", 0),
        @"sigmas": h3_super_link(@"sigmas", 0),
        @"latent_image": h3_super_link(@"state", 2)
    });
    nodes[@"separate"] = h3_super_node(@"LTXVSeparateAVLatent", @{
        @"av_latent": h3_super_link(@"denoise", 0)
    });
    nodes[@"crop_guides"] = h3_super_node(@"LTXVCropGuides", @{
        @"positive": h3_super_link(@"state", 0),
        @"negative": h3_super_link(@"state", 1),
        @"latent": h3_super_link(@"separate", 0)
    });
    nodes[@"save_state"] = h3_super_node(@"H3SuperSaveLatent", @{
        @"latent": h3_super_link(@"crop_guides", 2),
        @"filename": output_state_name
    });
    return nodes;
}

static NSDictionary *h3_super_decode_workflow(
        const h3_super_params *params, NSString *input_name,
        NSString *latent_name, NSString *prefix) {
    NSMutableDictionary *nodes = [NSMutableDictionary dictionary];
    NSArray *source_audio = nil;
    if (params->direct_handoff) {
        nodes[@"handoff_audio"] = h3_super_node(
            @"H3SuperLoadHandoffAudio", @{
                @"filename": input_name
            });
        source_audio = h3_super_link(@"handoff_audio", 0);
    } else {
        nodes[@"load_video"] = h3_super_node(@"LoadVideo", @{
            @"file": input_name
        });
        nodes[@"trim"] = h3_super_node(@"VideoFrameSample", @{
            @"video": h3_super_link(@"load_video", 0),
            @"num_frames": @(params->output_frames),
            @"strategy": @"head", @"seed": @0
        });
        nodes[@"components"] = h3_super_node(@"GetVideoComponents", @{
            @"video": h3_super_link(@"trim", 0)
        });
        source_audio = h3_super_link(@"components", 1);
    }
    nodes[@"latent"] = h3_super_node(@"H3SuperLoadLatent", @{
        @"filename": latent_name
    });
    if (params->decoder == H3_SUPER_DECODER_TAEHV) {
        nodes[@"decoded"] = h3_super_node(@"H3SuperTAEHVDecode", @{
            @"samples": h3_super_link(@"latent", 0),
            @"checkpoint": h3_super_string(params->taehv_checkpoint),
            @"compute_dtype": h3_super_string(params->taehv_compute_dtype),
            @"temporal_mode": h3_super_string(params->taehv_temporal_mode),
            @"keep_on_device": @(params->taehv_keep_on_device),
            @"expected_width": @(params->refined_width),
            @"expected_height": @(params->refined_height),
            @"expected_frames": @(params->output_frames)
        });
    } else {
        nodes[@"video_vae"] = h3_super_node(@"VAELoader", @{
            @"vae_name": h3_super_string(params->video_vae)
        });
        nodes[@"decoded"] = h3_super_node(@"VAEDecodeTiled", @{
            @"samples": h3_super_link(@"latent", 0),
            @"vae": h3_super_link(@"video_vae", 0),
            @"tile_size": @768, @"overlap": @64,
            @"temporal_size": @128, @"temporal_overlap": @24
        });
    }
    NSString *delivery_node = @"decoded";
    if (params->output_width != params->refined_width ||
        params->output_height != params->refined_height) {
        nodes[@"delivery_pixels"] = h3_super_node(@"ImageScale", @{
            @"image": h3_super_link(@"decoded", 0),
            @"upscale_method": @"lanczos",
            @"width": @(params->output_width),
            @"height": @(params->output_height), @"crop": @"center"
        });
        delivery_node = @"delivery_pixels";
    }
    nodes[@"video"] = h3_super_node(@"CreateVideo", @{
        @"images": h3_super_link(delivery_node, 0),
        @"fps": @(params->output_fps),
        @"audio": source_audio, @"bit_depth": @8
    });
    nodes[@"save"] = h3_super_node(@"SaveVideo", @{
        @"video": h3_super_link(@"video", 0),
        @"filename_prefix": prefix,
        @"format": @"mp4",
        /* SaveVideo's automatic H.264 path selected a roughly 1.1 Mb/s stream
         * for a 1344x768 five-second master.  That second lossy compression
         * measurably removes the already-limited high-frequency detail from
         * TAEHV.  Force a deterministic visually-near-lossless delivery encode
         * while the decoded frames are still available in the Comfy graph. */
        @"codec": @"h264",
        @"codec.encoding": @"re-encode",
        @"codec.encoding.crf": @18
    });
    return nodes;
}

static NSDictionary *h3_super_denoise_decode_workflow(
        const h3_super_params *params, NSString *input_state_name,
        NSString *input_name, NSString *prefix, uint64_t seed) {
    /*
     * Reuse the separately validated graph builders, but replace the disk
     * latent boundary with a direct graph edge. H3SuperTAEHVDecode unloads all
     * Comfy patchers before moving the cached TAEHV model to MPS, so the
     * Transformer and decoder are not intentionally device-resident together.
     */
    NSMutableDictionary *nodes = [h3_super_denoise_workflow(
        params, input_state_name, @"unused-fused-latent.pt", seed)
        mutableCopy];
    [nodes removeObjectForKey:@"save_state"];

    NSMutableDictionary *decode = [h3_super_decode_workflow(
        params, input_name, @"unused-fused-latent.pt", prefix)
        mutableCopy];
    [decode removeObjectForKey:@"latent"];
    decode[@"decoded"] = h3_super_node(@"H3SuperTAEHVDecode", @{
        @"samples": h3_super_link(@"crop_guides", 2),
        @"checkpoint": h3_super_string(params->taehv_checkpoint),
        @"compute_dtype": h3_super_string(params->taehv_compute_dtype),
        @"temporal_mode": h3_super_string(params->taehv_temporal_mode),
        @"keep_on_device": @(params->taehv_keep_on_device),
        @"expected_width": @(params->refined_width),
        @"expected_height": @(params->refined_height),
        @"expected_frames": @(params->output_frames)
    });
    for (NSString *key in decode) {
        if (nodes[key]) {
            return nil;
        }
        nodes[key] = decode[key];
    }
    return nodes;
}

static NSString *h3_super_backend_error(NSDictionary *record) {
    NSDictionary *status = record[@"status"];
    NSArray *messages = [status isKindOfClass:[NSDictionary class]] ?
        status[@"messages"] : nil;
    if (![messages isKindOfClass:[NSArray class]]) return @"unknown backend error";
    for (id item in [messages reverseObjectEnumerator]) {
        if (![item isKindOfClass:[NSArray class]] || [item count] < 2) continue;
        id payload = item[1];
        if ([payload isKindOfClass:[NSDictionary class]]) {
            NSString *message = payload[@"exception_message"];
            NSString *type = payload[@"exception_type"];
            if (message.length && type.length) {
                return [NSString stringWithFormat:@"%@: %@", type, message];
            }
            if (message.length) return message;
        }
    }
    return @"unknown backend error";
}

static NSDictionary *h3_super_history_record(const h3_super_params *params,
                                               NSString *prompt_id,
                                               double *queue_seconds,
                                               h3_super_result *result,
                                               char *error,
                                               size_t error_size) {
    double started = h3_super_now();
    double next_notice = started + 10.0;
    double next_memory_sample = started;
    while (h3_super_now() - started <= params->timeout_seconds) {
        NSString *path = [@"/history/" stringByAppendingString:prompt_id];
        NSInteger status = 0;
        NSData *data = h3_super_http(params, @"GET", path, nil, &status,
                                     error, error_size);
        if (!data) return nil;
        id object = h3_super_json_object(data, error, error_size);
        NSDictionary *record = [object isKindOfClass:[NSDictionary class]] ?
            [object objectForKey:prompt_id] : nil;
        if ([record isKindOfClass:[NSDictionary class]]) {
            NSDictionary *run_status = record[@"status"];
            NSNumber *completed = [run_status isKindOfClass:[NSDictionary class]] ?
                run_status[@"completed"] : nil;
            NSString *status_string = [run_status isKindOfClass:[NSDictionary class]] ?
                run_status[@"status_str"] : nil;
            if (completed.boolValue || [status_string isEqualToString:@"success"]) {
                if (queue_seconds) *queue_seconds = h3_super_now() - started;
                return record;
            }
            if ([status_string isEqualToString:@"error"]) {
                NSString *message = h3_super_backend_error(record);
                h3_super_fail(error, error_size, "LTX Stage 2 failed: %s",
                              message.UTF8String);
                return nil;
            }
        }
        double now = h3_super_now();
        if (now >= next_memory_sample && result) {
            uint64_t total = 0;
            uint64_t free_bytes = 0;
            if (h3_super_system_memory(params, &total, &free_bytes)) {
                result->backend_ram_total = total;
                if (!result->backend_ram_free_min ||
                    free_bytes < result->backend_ram_free_min) {
                    result->backend_ram_free_min = free_bytes;
                }
            }
            next_memory_sample = now + 5.0;
        }
        if (now >= next_notice) {
            fprintf(stderr, "h3: LTX Stage 2 running [%.0fs]\n", now - started);
            next_notice = now + 10.0;
        }
        double delay = params->poll_seconds;
        if (delay < 0.05) delay = 0.05;
        struct timespec sleep_value = {
            (time_t)delay,
            (long)((delay - (double)(time_t)delay) * 1000000000.0)
        };
        nanosleep(&sleep_value, NULL);
    }
    h3_super_fail(error, error_size,
                  "LTX Stage 2 timed out after %.0f seconds",
                  params->timeout_seconds);
    return nil;
}

static NSDictionary *h3_super_output_descriptor(NSDictionary *record) {
    NSDictionary *outputs = record[@"outputs"];
    NSDictionary *save = [outputs isKindOfClass:[NSDictionary class]] ?
        outputs[@"save"] : nil;
    if (![save isKindOfClass:[NSDictionary class]]) return nil;
    for (NSString *key in @[@"video", @"videos", @"images"]) {
        id value = save[key];
        if ([value isKindOfClass:[NSArray class]]) {
            for (id descriptor in value) {
                if ([descriptor isKindOfClass:[NSDictionary class]] &&
                    [descriptor[@"filename"] isKindOfClass:[NSString class]]) {
                    return descriptor;
                }
            }
        } else if ([value isKindOfClass:[NSDictionary class]] &&
                   [value[@"filename"] isKindOfClass:[NSString class]]) {
            return value;
        }
    }
    return nil;
}

static int h3_super_copy_output(NSString *source, NSString *destination,
                                uint64_t *bytes, char *error,
                                size_t error_size) {
    NSFileManager *manager = NSFileManager.defaultManager;
    NSString *parent = destination.stringByDeletingLastPathComponent;
    NSError *file_error = nil;
    if (parent.length && ![manager createDirectoryAtPath:parent
                             withIntermediateDirectories:YES
                                              attributes:nil error:&file_error]) {
        return h3_super_fail(error, error_size,
                             "cannot create output directory %s: %s",
                             parent.UTF8String,
                             file_error.localizedDescription.UTF8String);
    }
    NSString *temporary = [destination stringByAppendingFormat:@".partial-%d",
                            getpid()];
    [manager removeItemAtPath:temporary error:nil];
    if (![manager copyItemAtPath:source toPath:temporary error:&file_error]) {
        return h3_super_fail(error, error_size, "cannot copy Super output: %s",
                             file_error.localizedDescription.UTF8String);
    }
    [manager removeItemAtPath:destination error:nil];
    if (![manager moveItemAtPath:temporary toPath:destination error:&file_error]) {
        [manager removeItemAtPath:temporary error:nil];
        return h3_super_fail(error, error_size, "cannot install Super output: %s",
                             file_error.localizedDescription.UTF8String);
    }
    NSDictionary *attributes = [manager attributesOfItemAtPath:destination
                                                          error:&file_error];
    if (!attributes) {
        return h3_super_fail(error, error_size,
                             "cannot stat installed Super output: %s",
                             file_error.localizedDescription.UTF8String);
    }
    if (bytes) *bytes = attributes.fileSize;
    return 1;
}

static void h3_super_remove_state_files(const h3_super_params *params,
                                        NSString *state_files) {
    if (!params || !state_files.length) return;
    NSString *root = [h3_super_string(params->input_dir)
        stringByStandardizingPath];
    NSFileManager *manager = NSFileManager.defaultManager;
    for (NSString *name in [state_files componentsSeparatedByString:@","]) {
        if (!name.length || [name containsString:@"/"] ||
            [name containsString:@".."] || [name hasPrefix:@"."]) {
            fprintf(stderr, "h3: warning: refused unsafe Super state cleanup "
                    "name %s\n", name.UTF8String);
            continue;
        }
        NSString *path = [root stringByAppendingPathComponent:name];
        if (![manager fileExistsAtPath:path]) continue;
        NSError *remove_error = nil;
        if (![manager removeItemAtPath:path error:&remove_error]) {
            fprintf(stderr, "h3: warning: cannot remove Super state %s: %s\n",
                    path.UTF8String,
                    remove_error.localizedDescription.UTF8String);
        }
    }
}

static NSDictionary *h3_super_submit_phase(
        const h3_super_params *params, NSDictionary *workflow,
        NSString *identifier, NSString *label, h3_super_result *result,
        double *phase_seconds, char *error, size_t error_size) {
    NSDictionary *request = @{
        @"prompt": workflow,
        @"client_id": [NSString stringWithFormat:@"h3-super-%@-%@",
                       identifier, label]
    };
    NSData *body = h3_super_json_data(request, error, error_size);
    if (!body) return nil;
    double submitted_at = h3_super_now();
    NSInteger status = 0;
    NSData *response = h3_super_http(params, @"POST", @"/prompt", body,
                                     &status, error, error_size);
    if (!response) return nil;
    id response_object = h3_super_json_object(response, error, error_size);
    NSString *prompt_id = [response_object isKindOfClass:[NSDictionary class]] ?
        [response_object objectForKey:@"prompt_id"] : nil;
    if (![prompt_id isKindOfClass:[NSString class]] || !prompt_id.length) {
        h3_super_fail(error, error_size,
                      "ComfyUI did not return a prompt_id for phase %s",
                      label.UTF8String);
        return nil;
    }
    if (!h3_super_copy_string(result->prompt_id, sizeof(result->prompt_id),
                              prompt_id, error, error_size)) return nil;
    fprintf(stderr, "h3: submitted LTX Stage 2 %s phase (%s)\n",
            label.UTF8String, prompt_id.UTF8String);
    double queue_seconds = 0.0;
    NSDictionary *record = h3_super_history_record(
        params, prompt_id, &queue_seconds, result, error, error_size);
    result->queue_seconds += queue_seconds;
    if (phase_seconds) *phase_seconds = h3_super_now() - submitted_at;
    return record;
}

int h3_super_prefetch_transformer(const h3_super_params *params,
                                  double *seconds,
                                  char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (seconds) *seconds = 0.0;
    if (!params || !params->prefetch_transformer) {
        return h3_super_fail(error, error_size,
                             "Transformer prefetch is not enabled");
    }
    h3_super_result local = {0};
    double phase_seconds = 0.0;
    NSString *identifier = NSUUID.UUID.UUIDString.lowercaseString;
    NSDictionary *record = h3_super_submit_phase(
        params, h3_super_transformer_prefetch_workflow(params), identifier,
        @"transformer prefetch", &local, &phase_seconds, error, error_size);
    if (!record) return 0;
    if (seconds) *seconds = phase_seconds;
    return 1;
}

int h3_super_prefetch_conditioning(const h3_super_params *params,
                                   const char *prompt,
                                   h3_super_result *result,
                                   double *seconds,
                                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (seconds) *seconds = 0.0;
    if (!params || !params->prefetch_conditioning || !prompt || !*prompt ||
        !result) {
        return h3_super_fail(error, error_size,
                             "conditioning prefetch is not enabled or has "
                             "invalid arguments");
    }
    double started = h3_super_now();
    NSString *identifier = NSUUID.UUID.UUIDString.lowercaseString;
    NSString *state_name = [NSString stringWithFormat:
        @"h3-super-%@-conditioning.pt", identifier];
    if (!h3_super_copy_string(result->conditioning_prefetch_state,
                              sizeof(result->conditioning_prefetch_state),
                              state_name, error, error_size)) return 0;
    NSDictionary *record = h3_super_submit_phase(
        params, h3_super_conditioning_workflow(
            params, h3_super_string(prompt), state_name),
        identifier, @"conditioning prefetch", result, NULL,
        error, error_size);
    if (!record) return 0;
    if (params->prefetch_input_models) {
        result->input_model_prefetch_completed = 1;
    } else {
        double release_started = h3_super_now();
        if (!h3_super_release_backend(params, error, error_size)) return 0;
        result->backend_release_seconds += h3_super_now() - release_started;
    }
    if (seconds) *seconds = h3_super_now() - started;
    return 1;
}

int h3_super_prefetch_input_models(const h3_super_params *params,
                                   h3_super_result *result,
                                   double *seconds,
                                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (seconds) *seconds = 0.0;
    if (!params || !params->prefetch_input_models || !result) {
        return h3_super_fail(error, error_size,
                             "input-model prefetch is not enabled or has "
                             "invalid arguments");
    }
    double started = h3_super_now();
    NSString *identifier = NSUUID.UUID.UUIDString.lowercaseString;
    NSDictionary *record = h3_super_submit_phase(
        params, h3_super_input_models_prefetch_workflow(params), identifier,
        @"input-model prefetch", result, NULL, error, error_size);
    if (!record) {
        char ignored[256];
        h3_super_release_backend(params, ignored, sizeof(ignored));
        return 0;
    }
    /* Do not call /free here: this opt-in is useful only if the three small
     * input models remain device-resident until the immediately following
     * input prompt.  That phase performs the normal release before the 20 GiB
     * Transformer is loaded. */
    if (seconds) *seconds = h3_super_now() - started;
    return 1;
}

int h3_super_refine_comfy(const h3_super_params *params,
                          const char *stage1_path,
                          const char *prompt,
                          uint64_t seed,
                          const char *output_path,
                          h3_super_result *result,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!params || !stage1_path || !prompt || !*prompt ||
        !output_path || !*output_path || !result) {
        return h3_super_fail(error, error_size,
                             "invalid Super refinement arguments");
    }
    struct stat stage1_info;
    if (stat(stage1_path, &stage1_info) != 0 || !S_ISREG(stage1_info.st_mode) ||
        stage1_info.st_size <= 0) {
        return h3_super_fail(error, error_size,
                             "Stage-1 handoff is missing or empty: %s",
                             stage1_path);
    }
    NSString *input_root = [h3_super_string(params->input_dir)
        stringByStandardizingPath];
    NSString *stage1 = [h3_super_string(stage1_path) stringByStandardizingPath];
    NSString *relative = nil;
    NSString *prefix_with_slash = [input_root stringByAppendingString:@"/"];
    if ([stage1 hasPrefix:prefix_with_slash]) {
        relative = [stage1 substringFromIndex:prefix_with_slash.length];
    } else {
        return h3_super_fail(error, error_size,
                             "Stage-1 handoff must be written directly below the "
                             "configured ComfyUI input directory");
    }
    if ([relative containsString:@".."] || [relative hasPrefix:@"/"]) {
        return h3_super_fail(error, error_size,
                             "unsafe relative Stage-1 input path");
    }

    NSString *identifier = NSUUID.UUID.UUIDString.lowercaseString;
    NSString *backend_prefix = [@"video/h3-super-" stringByAppendingString:identifier];
    BOOL conditioning_prefetched = params->prefetch_conditioning &&
        result->conditioning_prefetch_completed &&
        result->conditioning_prefetch_state[0];
    BOOL input_models_prefetched = params->prefetch_input_models &&
        result->input_model_prefetch_completed;
    NSString *conditioning_name = conditioning_prefetched ?
        h3_super_string(result->conditioning_prefetch_state) :
        [NSString stringWithFormat:@"h3-super-%@-conditioning.pt", identifier];
    NSString *refiner_name = [NSString stringWithFormat:
        @"h3-super-%@-refiner.pt", identifier];
    NSString *latent_name = [NSString stringWithFormat:
        @"h3-super-%@-latent.pt", identifier];
    NSString *cleanup_names = params->fuse_denoise_decode ?
        [NSString stringWithFormat:@"%@,%@",
            conditioning_name, refiner_name] :
        [NSString stringWithFormat:@"%@,%@,%@",
            conditioning_name, refiner_name, latent_name];
    NSDictionary *descriptor = nil;
    NSString *filename = nil;
    NSString *subfolder = nil;
    NSString *backend_output = nil;
    NSString *opening_frame_name = nil;
    NSString *opening_frame_path = nil;
    double stage2_started = h3_super_now();
    if (!h3_super_prepare_first_frame(
            params, identifier, result, &opening_frame_name,
            &opening_frame_path, error, error_size)) {
        return 0;
    }
    NSDictionary *record = nil;
    if (!conditioning_prefetched) {
        record = h3_super_submit_phase(
            params, h3_super_conditioning_workflow(
                params, h3_super_string(prompt), conditioning_name),
            identifier, @"conditioning", result,
            &result->conditioning_seconds, error, error_size);
        if (!record) goto phase_failed;
    } else {
        result->conditioning_seconds = 0.0;
        fprintf(stderr,
                "h3: using conditioning artifact prefetched during H3 "
                "denoise (%s)\n", result->conditioning_prefetch_state);
    }
    double release_started = 0.0;
    if (!params->prefetch_transformer && !conditioning_prefetched) {
        release_started = h3_super_now();
        if (!h3_super_release_backend(params, error, error_size)) {
            goto phase_failed;
        }
        result->backend_release_seconds += h3_super_now() - release_started;
    }

    record = h3_super_submit_phase(
        params, h3_super_input_workflow(
            params, relative, conditioning_name, refiner_name,
            opening_frame_name,
            h3_super_string(result->first_frame_sha256),
            input_models_prefetched),
        identifier, @"input", result, &result->input_prepare_seconds,
        error, error_size);
    if (!record) goto phase_failed;
    if (opening_frame_path) {
        [NSFileManager.defaultManager removeItemAtPath:opening_frame_path
                                                error:nil];
        opening_frame_path = nil;
    }
    if (!params->prefetch_transformer) {
        release_started = h3_super_now();
        if (!h3_super_release_backend(params, error, error_size)) {
            goto phase_failed;
        }
        result->backend_release_seconds += h3_super_now() - release_started;
    }

    if (params->fuse_denoise_decode) {
        NSDictionary *workflow = h3_super_denoise_decode_workflow(
            params, refiner_name, relative, backend_prefix, seed);
        if (!workflow) {
            h3_super_fail(error, error_size,
                          "fused denoise/decode workflow has duplicate nodes");
            goto phase_failed;
        }
        record = h3_super_submit_phase(
            params, workflow, identifier, @"denoise+decode", result,
            &result->denoise_decode_seconds, error, error_size);
        if (!record) goto phase_failed;
    } else {
        record = h3_super_submit_phase(
            params, h3_super_denoise_workflow(
                params, refiner_name, latent_name, seed),
            identifier, @"denoise", result, &result->denoise_seconds,
            error, error_size);
        if (!record) goto phase_failed;
        release_started = h3_super_now();
        if (!h3_super_release_backend(params, error, error_size))
            goto phase_failed;
        result->backend_release_seconds += h3_super_now() - release_started;

        record = h3_super_submit_phase(
            params, h3_super_decode_workflow(
                params, relative, latent_name, backend_prefix),
            identifier, @"decode", result, &result->decode_seconds,
            error, error_size);
        if (!record) goto phase_failed;
    }
    result->stage2_seconds = h3_super_now() - stage2_started;
    descriptor = h3_super_output_descriptor(record);
    filename = descriptor[@"filename"];
    subfolder = descriptor[@"subfolder"];
    if (![filename isKindOfClass:[NSString class]] || !filename.length) {
        return h3_super_fail(error, error_size,
                             "ComfyUI history contains no SaveVideo output");
    }
    if (![subfolder isKindOfClass:[NSString class]]) subfolder = @"";
    if ([filename containsString:@"/"] || [filename containsString:@".."] ||
        [subfolder containsString:@".."] || [subfolder hasPrefix:@"/"]) {
        return h3_super_fail(error, error_size,
                             "ComfyUI returned an unsafe output path");
    }
    backend_output = h3_super_string(params->output_dir);
    if (subfolder.length) {
        backend_output = [backend_output stringByAppendingPathComponent:subfolder];
    }
    backend_output = [backend_output stringByAppendingPathComponent:filename];
    if (!h3_super_copy_string(result->backend_output,
                              sizeof(result->backend_output), backend_output,
                              error, error_size)) return 0;
    double copy_started = h3_super_now();
    if (!h3_super_copy_output(backend_output, h3_super_string(output_path),
                              &result->output_bytes, error, error_size)) return 0;
    result->output_copy_seconds = h3_super_now() - copy_started;
    release_started = h3_super_now();
    if (!h3_super_release_backend(params, error, error_size)) return 0;
    result->backend_release_seconds += h3_super_now() - release_started;
    double quality_started = h3_super_now();
    if (!h3_super_check_output_person_quality(
            params, output_path, result, error, error_size)) {
        result->output_quality_seconds = h3_super_now() - quality_started;
        if (unlink(output_path) != 0 && errno != ENOENT) {
            fprintf(stderr,
                    "h3: warning: cannot remove rejected refined output "
                    "%s: %s\n", output_path, strerror(errno));
        }
        goto phase_failed;
    }
    result->output_quality_seconds = h3_super_now() - quality_started;
    /* Keep conditioning/refiner/latent states recoverable until both the
     * encoder and the final quality gate have succeeded.  The old graph-side
     * cleanup ran before SaveVideo, so an encoder-argument failure discarded a
     * valid, expensive denoise result and forced a full Stage-2 replay. */
    h3_super_remove_state_files(params, cleanup_names);
    h3_super_system_memory(params, &result->backend_ram_total,
                           &result->backend_ram_free_after_release);
    uint64_t swapouts_after = 0;
    int after_available = h3_super_swapouts(&swapouts_after);
    result->swapouts_available = result->swapouts_available && after_available;
    if (after_available) result->system_swapouts_after = swapouts_after;
    result->no_swap_gate_passed = result->swapouts_available &&
        result->system_swapouts_after == result->system_swapouts_before;
    return 1;

phase_failed: {
        char ignored[256];
        if (opening_frame_path) {
            [NSFileManager.defaultManager removeItemAtPath:opening_frame_path
                                                    error:nil];
        }
        h3_super_release_backend(params, ignored, sizeof(ignored));
        return 0;
    }
}

static NSDictionary *h3_super_stage1_provenance(
        const h3_super_params *params) {
    if (!params || !params->stage1_model_dir ||
        !*params->stage1_model_dir) {
        return @{
            @"validated": @NO,
            @"status": @"manifest_not_configured"
        };
    }
    NSString *root = h3_super_string(params->stage1_model_dir);
    NSString *path = [root stringByAppendingPathComponent:
        @"FL2VA/transformer/h3-turbo-merge-manifest.json"];
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data.length) {
        return @{
            @"validated": @NO,
            @"status": @"manifest_missing",
            @"manifest_path": path
        };
    }
    NSError *json_error = nil;
    id object = [NSJSONSerialization JSONObjectWithData:data
                                                options:0
                                                  error:&json_error];
    if (![object isKindOfClass:NSDictionary.class]) {
        return @{
            @"validated": @NO,
            @"status": @"manifest_invalid_json",
            @"manifest_path": path
        };
    }
    NSDictionary *manifest = object;
    NSDictionary *identity = [manifest[@"identity"]
        isKindOfClass:NSDictionary.class] ? manifest[@"identity"] : nil;
    NSDictionary *mapping = [manifest[@"mapping"]
        isKindOfClass:NSDictionary.class] ? manifest[@"mapping"] : nil;
    NSDictionary *source = [manifest[@"source"]
        isKindOfClass:NSDictionary.class] ? manifest[@"source"] : nil;
    NSDictionary *shards = [manifest[@"shards"]
        isKindOfClass:NSDictionary.class] ? manifest[@"shards"] : nil;
    NSNumber *strength = [identity[@"strength"]
        isKindOfClass:NSNumber.class] ? identity[@"strength"] : nil;
    BOOL common_valid =
        [manifest[@"schema"] isEqual:@"h3-turbo-merge-manifest-v2"] &&
        [identity[@"profile"] isEqual:@"lightx2v-4step"] &&
        [identity[@"raw_tensors"] integerValue] == 624 &&
        [identity[@"raw_pairs"] integerValue] == 312 &&
        [identity[@"mapped_weights"] integerValue] == 208 &&
        [source[@"repository"] isEqual:@"lightx2v/Minimax-h3-Turbo"] &&
        [mapping[@"raw_tensors"] integerValue] == 624 &&
        [mapping[@"raw_pairs"] integerValue] == 312 &&
        [mapping[@"mapped_weights"] integerValue] == 208 &&
        shards.count == 13;

    NSString *variant = nil;
    NSString *status = nil;
    NSString *revision = nil;
    NSString *lora_sha256 = nil;
    NSString *training_resolution = nil;
    NSString *contract_tag = nil;
    unsigned long long lora_bytes = 0;
    int recommended_steps = 0;
    double expected_strength = 0.0;
    double video_flow_shift = 0.0;
    double audio_flow_shift = 0.0;
    BOOL sana_v2_stage1_exact = NO;

    BOOL v01_optional_contract =
        (!identity[@"variant"] ||
         [identity[@"variant"] isEqual:@"v0.1-544p"]) &&
        (!identity[@"video_flow_shift"] ||
         fabs([identity[@"video_flow_shift"] doubleValue] - 12.0) <= 1e-12) &&
        (!identity[@"audio_flow_shift"] ||
         fabs([identity[@"audio_flow_shift"] doubleValue] - 3.0) <= 1e-12) &&
        (!identity[@"training_resolution"] ||
         [identity[@"training_resolution"]
            isEqual:@"544p_mixed_aspect_ratio"]) &&
        (!identity[@"recommended_steps"] ||
         [identity[@"recommended_steps"] integerValue] == 4) &&
        (!source[@"artifact"] ||
         [source[@"artifact"] isEqual:@"v0.1-544p"]);
    BOOL v01 = common_valid && strength && v01_optional_contract &&
        [identity[@"source_revision"]
            isEqual:@"050494d5fe05bd1b1140b8565ea51dc33a5085a5"] &&
        [identity[@"lora_sha256"]
            isEqual:@"5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29"] &&
        [identity[@"lora_bytes"] unsignedLongLongValue] == 1383677888ULL &&
        fabs(strength.doubleValue - 0.0625) <= 1e-12 &&
        [source[@"revision"]
            isEqual:@"050494d5fe05bd1b1140b8565ea51dc33a5085a5"];
    BOOL v11 = common_valid && strength &&
        [identity[@"source_revision"]
            isEqual:@"2f8ea0dc0a7e2b26c9a43124eb89673787189b4e"] &&
        [identity[@"lora_sha256"]
            isEqual:@"b5e25a59292d51bca3fc02b9a0b2284e11b4eb20921a9c5adc2db785956b8966"] &&
        [identity[@"lora_bytes"] unsignedLongLongValue] == 1383677808ULL &&
        fabs(strength.doubleValue - 1.0) <= 1e-12 &&
        [identity[@"variant"] isEqual:@"v1.1-768p"] &&
        fabs([identity[@"video_flow_shift"] doubleValue] - 6.0) <= 1e-12 &&
        fabs([identity[@"audio_flow_shift"] doubleValue] - 3.0) <= 1e-12 &&
        [identity[@"training_resolution"] isEqual:@"768p"] &&
        [identity[@"recommended_steps"] integerValue] == 4 &&
        [source[@"revision"]
            isEqual:@"2f8ea0dc0a7e2b26c9a43124eb89673787189b4e"] &&
        [source[@"artifact"] isEqual:@"v1.1-768p"];
    BOOL v10_8step = common_valid && strength &&
        [identity[@"source_revision"]
            isEqual:@"5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1"] &&
        [identity[@"lora_sha256"]
            isEqual:@"e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"] &&
        [identity[@"lora_bytes"] unsignedLongLongValue] == 1383677768ULL &&
        fabs(strength.doubleValue - 0.0625) <= 1e-12 &&
        [identity[@"variant"] isEqual:@"v1.0-544p-8step"] &&
        fabs([identity[@"video_flow_shift"] doubleValue] - 12.0) <= 1e-12 &&
        fabs([identity[@"audio_flow_shift"] doubleValue] - 3.0) <= 1e-12 &&
        [identity[@"training_resolution"]
            isEqual:@"544p_mixed_aspect_ratio"] &&
        [identity[@"recommended_steps"] integerValue] == 8 &&
        [source[@"revision"]
            isEqual:@"5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1"] &&
        [source[@"artifact"] isEqual:@"v1.0-544p-8step"];
    if (v01) {
        variant = @"v0.1-544p";
        status = @"sana_lightx2v_merge_manifest_validated";
        revision = @"050494d5fe05bd1b1140b8565ea51dc33a5085a5";
        lora_sha256 =
            @"5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29";
        training_resolution = @"544p_mixed_aspect_ratio";
        lora_bytes = 1383677888ULL;
        expected_strength = 0.0625;
        video_flow_shift = 12.0;
        audio_flow_shift = 3.0;
        recommended_steps = 4;
        contract_tag = @"sana_lightx2v_v01_544p";
        sana_v2_stage1_exact = YES;
    } else if (v11) {
        variant = @"v1.1-768p";
        status = @"lightx2v_v11_768p_merge_manifest_validated";
        revision = @"2f8ea0dc0a7e2b26c9a43124eb89673787189b4e";
        lora_sha256 =
            @"b5e25a59292d51bca3fc02b9a0b2284e11b4eb20921a9c5adc2db785956b8966";
        training_resolution = @"768p";
        lora_bytes = 1383677808ULL;
        expected_strength = 1.0;
        video_flow_shift = 6.0;
        audio_flow_shift = 3.0;
        recommended_steps = 4;
        contract_tag = @"lightx2v_v11_768p";
    } else if (v10_8step) {
        variant = @"v1.0-544p-8step";
        status = @"lightx2v_v10_544p_8step_merge_manifest_validated";
        revision = @"5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1";
        lora_sha256 =
            @"e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f";
        training_resolution = @"544p_mixed_aspect_ratio";
        lora_bytes = 1383677768ULL;
        expected_strength = 0.0625;
        video_flow_shift = 12.0;
        audio_flow_shift = 3.0;
        recommended_steps = 8;
        contract_tag = @"lightx2v_v10_544p_8step";
    }
    BOOL valid = v01 || v11 || v10_8step;

    char manifest_sha256[65] = "";
    if (!h3_super_sha256_file(path.fileSystemRepresentation,
                              manifest_sha256, NULL, NULL, 0)) {
        manifest_sha256[0] = '\0';
        valid = NO;
    }
    if (!valid) {
        return @{
            @"validated": @NO,
            @"status": @"manifest_contract_mismatch",
            @"manifest_path": path,
            @"manifest_sha256": h3_super_string(manifest_sha256)
        };
    }
    return @{
        @"validated": @YES,
        @"status": status,
        @"manifest_path": path,
        @"manifest_sha256": h3_super_string(manifest_sha256),
        @"repository": @"lightx2v/Minimax-h3-Turbo",
        @"revision": revision,
        @"variant": variant,
        @"contract_tag": contract_tag,
        @"sana_v2_stage1_exact": @(sana_v2_stage1_exact),
        @"lora_sha256": lora_sha256,
        @"lora_bytes": @(lora_bytes),
        @"strength": @(expected_strength),
        @"video_flow_shift": @(video_flow_shift),
        @"audio_flow_shift": @(audio_flow_shift),
        @"training_resolution": training_resolution,
        @"recommended_steps": @(recommended_steps),
        @"raw_tensors": @624,
        @"raw_pairs": @312,
        @"mapped_weights": @208,
        @"shards": @13,
        @"shard_hashes_recorded": @YES,
        @"runtime_shard_hash_recheck": @NO
    };
}

int h3_super_stage1_quality_contract(const h3_super_params *params,
                                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    NSDictionary *provenance = h3_super_stage1_provenance(params);
    if (![provenance[@"validated"] boolValue] ||
        ![provenance[@"variant"] isEqual:@"v1.1-768p"] ||
        ![provenance[@"revision"]
            isEqual:@"2f8ea0dc0a7e2b26c9a43124eb89673787189b4e"] ||
        fabs([provenance[@"strength"] doubleValue] - 1.0) > 1e-12 ||
        fabs([provenance[@"video_flow_shift"] doubleValue] - 6.0) > 1e-12 ||
        fabs([provenance[@"audio_flow_shift"] doubleValue] - 3.0) > 1e-12) {
        NSString *status_value = provenance[@"status"];
        const char *status = [status_value isKindOfClass:NSString.class] ?
            status_value.UTF8String : "unknown";
        return h3_super_fail(
            error, error_size,
            "person-strict-v1 requires the pinned LightX2V v1.1 768p "
            "Stage-1 merge contract (%s)", status);
    }
    return 1;
}

int h3_super_stage1_schedule(const h3_super_params *params,
                             double *video_shift,
                             double *audio_shift,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!video_shift || !audio_shift) {
        return h3_super_fail(error, error_size,
                             "Super Stage-1 schedule outputs are NULL");
    }
    NSDictionary *provenance = h3_super_stage1_provenance(params);
    if (![provenance[@"validated"] boolValue]) {
        NSString *status_value = provenance[@"status"];
        const char *status = [status_value isKindOfClass:NSString.class] ?
            status_value.UTF8String : "unknown";
        return h3_super_fail(
            error, error_size,
            "LightX2V merge manifest is not a supported pinned contract (%s)",
            status);
    }
    NSNumber *video = provenance[@"video_flow_shift"];
    NSNumber *audio = provenance[@"audio_flow_shift"];
    if (![video isKindOfClass:NSNumber.class] ||
        ![audio isKindOfClass:NSNumber.class] ||
        !isfinite(video.doubleValue) || !isfinite(audio.doubleValue) ||
        video.doubleValue <= 0.0 || audio.doubleValue <= 0.0) {
        return h3_super_fail(error, error_size,
                             "validated Stage-1 manifest has invalid shifts");
    }
    *video_shift = video.doubleValue;
    *audio_shift = audio.doubleValue;
    return 1;
}

static int h3_super_handoff_version(const char *path) {
    if (!path || !*path) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    char header[H3_SUPER_HANDOFF_HEADER_BYTES + 1];
    size_t bytes = fread(header, 1, H3_SUPER_HANDOFF_HEADER_BYTES, file);
    int read_error = ferror(file);
    fclose(file);
    if (read_error) return 0;
    header[bytes] = '\0';
    if (strstr(header, "\"schema\":\"h3-super-handoff-v2\"")) return 2;
    if (strstr(header, "\"schema\":\"h3-super-handoff-v1\"")) return 1;
    return 0;
}

int h3_super_write_telemetry(const h3_super_params *params,
                             const h3_super_result *result,
                             const char *stage1_path,
                             const char *output_path,
                             uint64_t seed,
                             double stage1_seconds,
                             double total_seconds,
                             char *error, size_t error_size) {
    if (!params || !result || !params->telemetry_path ||
        !*params->telemetry_path) return 1;
    NSDictionary *turbo_provenance = h3_super_stage1_provenance(params);
    BOOL lightx2v_stage1 = [turbo_provenance[@"validated"] boolValue];
    NSString *stage1_variant = [turbo_provenance[@"variant"]
        isKindOfClass:NSString.class] ? turbo_provenance[@"variant"] : nil;
    BOOL sana_v2_stage1_exact =
        [turbo_provenance[@"sana_v2_stage1_exact"] boolValue];
    BOOL runtime_official_stage2 = params->refiner_lora &&
        h3_super_uses_runtime_refiner_base(params) &&
        !strcmp(params->refiner_lora, h3_super_official_refiner_lora) &&
        fabs(params->refiner_lora_strength - 0.8) <= 1e-12;
    BOOL bf16_runtime_stage2 = runtime_official_stage2 &&
        h3_super_uses_bf16_transformer(params);
    BOOL merged_official_stage2 = h3_super_uses_merged_refiner(params);
    BOOL hybrid_official_stage2 = h3_super_uses_hybrid_refiner(params);
    BOOL official_stage2 = runtime_official_stage2 || merged_official_stage2;
    const char *official_transformer_sha256 =
        h3_super_official_transformer_sha256;
    unsigned long long official_transformer_bytes = 21504034224ULL;
    if (hybrid_official_stage2) {
        official_transformer_sha256 =
            h3_super_official_hybrid_refiner_sha256;
        official_transformer_bytes =
            h3_super_official_hybrid_refiner_bytes;
    } else if (merged_official_stage2) {
        official_transformer_sha256 =
            h3_super_official_merged_refiner_sha256;
    } else if (bf16_runtime_stage2) {
        official_transformer_sha256 =
            h3_super_official_bf16_transformer_sha256;
        official_transformer_bytes =
            h3_super_official_bf16_transformer_bytes;
    }
    BOOL default_stage2_schedule =
        params->stage2_schedule == H3_SUPER_STAGE2_SCHEDULE_DEFAULT;
    BOOL stage1_admitted = result->stage1_quality_attempts == 0 ||
        result->stage1_quality_attempts > result->stage1_quality_rejections;
    BOOL refine_admitted = result->refine_quality_attempts > 0 &&
        result->refine_quality_attempts > result->refine_quality_rejections;
    BOOL output_rejected =
        (result->refine_quality_attempts > 0 && !refine_admitted) ||
        (result->output_quality_gate_applied &&
         !result->output_quality_gate_passed);
    NSString *run_status = !stage1_admitted ? @"stage1_rejected" :
        output_rejected ? @"output_rejected" : @"complete";
    int handoff_version = params->direct_handoff ?
        h3_super_handoff_version(stage1_path) : 0;
    BOOL prepared_bf16_handoff = handoff_version == 2;
    NSString *direct_handoff_tag = prepared_bf16_handoff ?
        @"direct-bf16" : @"direct-f32";
    NSString *execution_tag = @"staged";
    if (params->prefetch_transformer) {
        execution_tag = params->prefetch_start_block ?
            [NSString stringWithFormat:@"prefetch-block%d-evict%d",
                params->prefetch_start_block, params->final_evict_blocks] :
            params->prefetch_start_step ?
                params->final_evict_blocks ?
                    [NSString stringWithFormat:@"prefetch-step%d-evict%d",
                        params->prefetch_start_step,
                        params->final_evict_blocks] :
                    [NSString stringWithFormat:@"prefetch-step%d",
                        params->prefetch_start_step] : @"prefetch";
    } else if (params->prefetch_conditioning) {
        execution_tag = params->prefetch_start_block ?
            [NSString stringWithFormat:
                @"conditioning-prefetch-block%d-evict%d",
                params->prefetch_start_block, params->final_evict_blocks] :
            params->prefetch_start_step ?
                params->final_evict_blocks ?
                    [NSString stringWithFormat:
                        @"conditioning-prefetch-step%d-evict%d",
                        params->prefetch_start_step,
                        params->final_evict_blocks] :
                    [NSString stringWithFormat:
                        @"conditioning-prefetch-step%d",
                        params->prefetch_start_step] :
                @"conditioning-prefetch";
    }
    if (params->prefetch_input_models) {
        execution_tag = [execution_tag stringByAppendingString:@"-inputmodels"];
    }
    if (params->fuse_denoise_decode) {
        execution_tag = [execution_tag stringByAppendingString:@"-fuseddecode"];
    }
    NSString *contract = nil;
    if (params->direct_handoff) {
        NSString *variant = @"balanced";
        if (params->profile == H3_SUPER_PROFILE_480P_FAST) variant = @"fast";
        else if (params->profile == H3_SUPER_PROFILE_480P_QUALITY)
            variant = @"quality";
        else if (params->profile == H3_SUPER_PROFILE_480P_MOTION)
            variant = @"motion";
        else if (params->profile == H3_SUPER_PROFILE_V2) variant = @"video-v2";
        if (lightx2v_stage1 && merged_official_stage2) {
            if (h3_super_uses_hybrid_refiner(params) &&
                default_stage2_schedule) {
                contract = [NSString stringWithFormat:
                    params->first_frame ?
                    @"local-super-480p-v14-%@-%@-%@-lightx2v-official-ltx-refiner-video-self-bf16-original-guide-comfy-%@" :
                    @"local-super-480p-v14-%@-%@-%@-lightx2v-official-ltx-refiner-video-self-bf16-comfy-%@",
                    variant,
                    execution_tag,
                    params->sol_attention ? @"sol" : @"dense",
                    direct_handoff_tag];
            } else if (h3_super_uses_hybrid_refiner(params)) {
                contract = [NSString stringWithFormat:
                    params->first_frame ?
                    @"local-super-480p-v15-%@-%@-%@-skip0p9-lightx2v-official-ltx-refiner-video-self-bf16-original-guide-comfy-%@" :
                    @"local-super-480p-v15-%@-%@-%@-skip0p9-lightx2v-official-ltx-refiner-video-self-bf16-comfy-%@",
                    variant,
                    execution_tag,
                    params->sol_attention ? @"sol" : @"dense",
                    direct_handoff_tag];
            } else if (default_stage2_schedule) {
                contract = [NSString stringWithFormat:
                    params->first_frame ?
                    @"local-super-480p-v%d-%@-%@-%@-lightx2v-official-ltx-refiner-merged-int8-original-guide-comfy-%@" :
                    @"local-super-480p-v%d-%@-%@-%@-lightx2v-official-ltx-refiner-merged-int8-comfy-%@",
                    prepared_bf16_handoff ? 12 : 10,
                    variant,
                    execution_tag,
                    params->sol_attention ? @"sol" : @"dense",
                    direct_handoff_tag];
            } else {
                contract = [NSString stringWithFormat:
                    params->first_frame ?
                    @"local-super-480p-v%d-%@-%@-%@-skip0p9-lightx2v-official-ltx-refiner-merged-int8-original-guide-comfy-%@" :
                    @"local-super-480p-v%d-%@-%@-%@-skip0p9-lightx2v-official-ltx-refiner-merged-int8-comfy-%@",
                    prepared_bf16_handoff ? 13 : 11,
                    variant,
                    execution_tag,
                    params->sol_attention ? @"sol" : @"dense",
                    direct_handoff_tag];
            }
        } else if (lightx2v_stage1 && runtime_official_stage2) {
            if (bf16_runtime_stage2) {
                contract = [NSString stringWithFormat:
                    params->first_frame ?
                    @"local-super-480p-v18-%@-%@-%@-lightx2v-official-bf16-ltx-refiner-bypass-original-guide-comfy-%@" :
                    @"local-super-480p-v18-%@-%@-%@-lightx2v-official-bf16-ltx-refiner-bypass-comfy-%@",
                    variant,
                    execution_tag,
                    params->sol_attention ? @"sol" : @"dense",
                    direct_handoff_tag];
            } else {
                contract = [NSString stringWithFormat:
                    params->first_frame ?
                    @"local-super-480p-v17-%@-%@-%@-lightx2v-official-ltx-refiner-bypass-original-guide-comfy-%@" :
                    @"local-super-480p-v17-%@-%@-%@-lightx2v-official-ltx-refiner-bypass-comfy-%@",
                    variant,
                    execution_tag,
                    params->sol_attention ? @"sol" : @"dense",
                    direct_handoff_tag];
            }
        } else if (lightx2v_stage1) {
            contract = [NSString stringWithFormat:
                params->first_frame ?
                @"local-super-480p-v8-%@-%@-%@-lightx2v-original-guide-comfy-%@" :
                @"local-super-480p-v8-%@-%@-%@-lightx2v-comfy-%@",
                variant,
                execution_tag,
                params->sol_attention ? @"sol" : @"dense",
                direct_handoff_tag];
        } else {
            contract = [NSString stringWithFormat:
                params->first_frame ?
                @"local-super-480p-v7-%@-%@-%@-original-guide-comfy-%@" :
                @"local-super-480p-v6-%@-%@-%@-comfy-%@",
                variant,
                execution_tag,
                params->sol_attention ? @"sol" : @"dense",
                direct_handoff_tag];
        }
    } else switch (params->profile) {
        case H3_SUPER_PROFILE_480P:
            contract = [NSString stringWithFormat:
                params->sol_attention ?
                    @"local-super-480p-v5-balanced-%@-sol-comfy-mp4" :
                    @"local-super-480p-v4-balanced-%@-comfy-mp4",
                execution_tag];
            break;
        case H3_SUPER_PROFILE_480P_FAST:
            contract = [NSString stringWithFormat:
                params->sol_attention ?
                    @"local-super-480p-v5-fast-%@-sol-comfy-mp4" :
                    @"local-super-480p-v4-fast-%@-comfy-mp4",
                execution_tag];
            break;
        case H3_SUPER_PROFILE_480P_QUALITY:
            contract = [NSString stringWithFormat:
                params->sol_attention ?
                    @"local-super-480p-v5-quality-%@-sol-comfy-mp4" :
                    @"local-super-480p-v4-quality-%@-comfy-mp4",
                execution_tag];
            break;
        case H3_SUPER_PROFILE_480P_MOTION:
            contract = [NSString stringWithFormat:
                params->sol_attention ?
                    @"local-super-480p-v5-motion-%@-sol-comfy-mp4" :
                    @"local-super-480p-v4-motion-%@-comfy-mp4",
                execution_tag];
            break;
        case H3_SUPER_PROFILE_V2:
            contract = params->sol_attention ?
                @"local-super-video-v2-sol-comfy-mp4" :
                @"local-super-video-v1-comfy-mp4";
            break;
    }
    if (params->stage2_schedule != H3_SUPER_STAGE2_SCHEDULE_DEFAULT) {
        NSString *schedule_tag =
            params->stage2_schedule == H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9 ?
                @"-skip0p9" : @"-tail0p42";
        if ([contract rangeOfString:schedule_tag].location == NSNotFound) {
            contract = [contract stringByAppendingString:schedule_tag];
        }
    }
    if (params->stage1_steps != 4) {
        contract = [contract stringByAppendingFormat:@"-stage1-s%d",
                    params->stage1_steps];
    }
    if (params->strict_person_quality) {
        contract = [contract stringByAppendingString:@"-person-strict-v1"];
    }
    if (params->person_quality_gate) {
        contract = [contract stringByAppendingString:@"-persongate"];
    } else if (params->face_quality_gate) {
        contract = [contract stringByAppendingString:@"-facegate"];
    }
    if (params->min_face_area > 0.0) {
        contract = [contract stringByAppendingFormat:@"-minface%.4f",
                    params->min_face_area];
    }
    if (params->quality_attempts > 1) {
        contract = [contract stringByAppendingFormat:@"-attempts%d",
                    params->quality_attempts];
    }
    if (params->person_quality_gate) {
        contract = [contract stringByAppendingString:@"-postgate"];
    } else if (params->output_quality_gate) {
        contract = [contract stringByAppendingString:@"-outputgate"];
    }
    contract = [contract stringByAppendingString:params->first_frame ?
        @"-sana-guide-crf18-singlepass" : @"-sana-singlepass"];
    contract = [contract stringByAppendingString:@"-h264crf18"];
    NSString *decoder = params->decoder == H3_SUPER_DECODER_TAEHV ?
        @"TAEHV wide" : @"official tiled Video VAE";
    NSString *phase_topology = params->fuse_denoise_decode ?
        @"three phase prompts; denoise and TAEHV decode share one prompt" :
        @"four phase prompts with executor reset between phases";
    NSString *execution_topology = phase_topology;
    NSString *conditioning_residency = params->prefetch_input_models ?
        @"conditioning prompt tail replaces Gemma residency with fully loaded Video VAE, Audio VAE, and latent upscaler until Stage 2 input" :
        @"compact CPU artifact retained; all Comfy models released before Stage 2 input";
    if (params->prefetch_transformer) {
        execution_topology = params->prefetch_start_block ?
            [NSString stringWithFormat:
                @"completion-safe H3 final-pass eviction in %d-block groups; background Transformer prefetch after block %d/50 plus %@; conditioning/input retain residency",
                params->final_evict_blocks, params->prefetch_start_block,
                phase_topology] :
            params->final_evict_blocks ?
                [NSString stringWithFormat:
                    @"background Transformer prefetch after H3 denoise %d/%d; completion-safe H3 final-pass eviction in %d-block groups; %@; conditioning/input retain residency",
                    params->prefetch_start_step,
                    params->stage1_steps,
                    params->final_evict_blocks, phase_topology] :
                [NSString stringWithFormat:
                    @"background Transformer prefetch after H3 denoise %d/%d plus %@; conditioning/input retain residency",
                    params->prefetch_start_step, params->stage1_steps,
                    phase_topology];
    } else if (params->prefetch_conditioning) {
        execution_topology = params->prefetch_start_block ?
            [NSString stringWithFormat:
                @"completion-safe H3 final-pass eviction in %d-block groups; background Gemma conditioning prefetch after block %d/50; %@",
                params->final_evict_blocks, params->prefetch_start_block,
                conditioning_residency] :
            params->final_evict_blocks ?
                [NSString stringWithFormat:
                    @"background Gemma conditioning prefetch after H3 denoise %d/%d; completion-safe H3 final-pass eviction in %d-block groups; %@",
                    params->prefetch_start_step,
                    params->stage1_steps,
                    params->final_evict_blocks, conditioning_residency] :
                [NSString stringWithFormat:
                    @"background Gemma conditioning prefetch after H3 denoise %d/%d; %@",
                    params->prefetch_start_step, params->stage1_steps,
                    conditioning_residency];
    } else if (params->final_evict_blocks) {
        execution_topology = [NSString stringWithFormat:
            @"completion-safe H3 final-pass eviction in %d-block groups followed by staged LTX execution",
            params->final_evict_blocks];
    }
    if (params->fuse_denoise_decode) {
        execution_topology = [execution_topology stringByAppendingString:
            @"; TAEHV unloads the Transformer in-graph before decoder residency; no latent file or intermediate /free boundary"];
    }
    NSMutableArray<NSString *> *differences = [NSMutableArray arrayWithObject:
        @"ComfyUI component offload instead of native h3.c LTX block streaming"];
    if (!official_stage2) {
        [differences addObject:
            @"local INT8 distilled LTX Transformer identity is not proven equal to Sana's BF16 base plus refiner LoRA at strength 0.8"];
    }
    if (!lightx2v_stage1) {
        [differences addObject:
            @"local merged Turbo checkpoint identity is not proven equal to Sana LightX2V LoRA scale 0.0625"];
    } else if ([stage1_variant isEqual:@"v1.1-768p"]) {
        [differences addObject:
            @"LightX2V v1.1 768p Stage 1 replaces Sana Super v2's v0.1 544p adapter and uses its published strength 1.0 plus video/audio flow shifts 6/3"];
    } else if ([stage1_variant isEqual:@"v1.0-544p-8step"]) {
        [differences addObject:
            @"LightX2V v1.0 544p eight-step Stage 1 replaces Sana Super v2's four-step v0.1 adapter and uses eight H3 forwards"];
    } else if (!sana_v2_stage1_exact) {
        [differences addObject:
            @"validated LightX2V Stage 1 differs from Sana Super v2's pinned v0.1 four-step adapter"];
    }
    if (!params->first_frame) {
        [differences addObject:
            @"text-only run uses the decoded Stage-1 first frame instead of Sana's SHA-pinned original opening-frame asset"];
    }
    if (params->direct_handoff) {
        [differences addObject:prepared_bf16_handoff ?
            @"file-backed BF16 tensor instead of Sana's pinned BF16 TCP tensor" :
            @"mmap F32 video artifact instead of Sana's pinned BF16 TCP tensor"];
    } else {
        [differences addObject:
            @"MP4 handoff instead of direct BF16/FP32 tensors"];
    }
    if (!params->sol_attention) {
        [differences addObject:
            @"dense attention instead of layer-0 dense plus layers-1-47 Sol-Attn"];
    } else {
        [differences addObject:
            @"Apple Metal Sol-Attn instead of Sana's CUDA architecture-dispatched kernel"];
    }
    if (params->decoder != H3_SUPER_DECODER_TAEHV) {
        [differences addObject:@"official Video VAE decode instead of TAEHV wide"];
    }
    if (params->profile != H3_SUPER_PROFILE_V2) {
        [differences addObject:[NSString stringWithFormat:
            @"480p delivery profile uses a %dx%d learned-x2 canvas followed by a center resize to %dx%d",
            params->refined_width, params->refined_height,
            params->output_width, params->output_height]];
    }
    if (!default_stage2_schedule) {
        int updates = h3_super_stage2_updates(params->stage2_schedule);
        [differences addObject:[NSString stringWithFormat:
            @"diagnostic %@ Stage-2 schedule uses %d LTX update%@ instead of Sana Super v2's fixed three-update schedule",
            h3_super_string(h3_super_stage2_schedule_name(
                params->stage2_schedule)),
            updates, updates == 1 ? @"" : @"s"]];
    }
    if (params->stage1_steps != 4) {
        [differences addObject:[NSString stringWithFormat:
            @"diagnostic H3 Stage-1 schedule uses %d denoising passes instead of Sana Super v2's fixed four-pass LightX2V schedule",
            params->stage1_steps]];
    }
    NSDictionary *payload = @{
        @"schema": @"h3-super-telemetry-v17",
        @"status": run_status,
        @"contract": contract,
        @"profile": h3_super_string(h3_super_profile_name(params->profile)),
        @"quality_preset": params->strict_person_quality ?
            @"person-strict-v1" : @"custom",
        @"strict_sana_v2_parity": @NO,
        @"backend": @"comfyui-mps",
        @"endpoint": h3_super_string(params->endpoint),
        @"prompt_id": h3_super_string(result->prompt_id),
        @"seed": @(seed),
        @"requested_seed": @(result->stage1_quality_attempts > 0 ?
            result->stage1_first_seed : seed),
        @"stage1": @{
            @"width": @(params->stage1_width),
            @"height": @(params->stage1_height),
            @"frames": @(params->stage1_frames),
            @"fps": @(params->output_fps),
            @"steps": @(params->stage1_steps), @"decoder": @"TAEH3",
            @"conditioning": params->first_frame ?
                @"FL2VA first-frame" : @"text-only",
            @"handoff": params->direct_handoff ?
                (prepared_bf16_handoff ?
                    @"mmap pre-resized BF16 video + FP32 PCM" :
                    @"mmap F32 video + FP32 PCM") : @"H.264/AAC MP4",
            @"handoff_schema": params->direct_handoff ?
                (prepared_bf16_handoff ? @"h3-super-handoff-v2" :
                    @"h3-super-handoff-v1") : @"mp4",
            @"handoff_video_dtype": prepared_bf16_handoff ?
                @"bfloat16" : (params->direct_handoff ? @"float32" : @"uint8"),
            @"handoff_video_width": @(prepared_bf16_handoff ?
                params->input_width : params->stage1_width),
            @"handoff_video_height": @(prepared_bf16_handoff ?
                params->input_height : params->stage1_height),
            @"handoff_bytes": @(result->handoff_bytes),
            @"handoff_seconds": @(result->handoff_seconds),
            @"face_quality_gate": @{
                @"enabled": @(
                    params->face_quality_gate || params->person_quality_gate),
                @"applied": @(result->face_quality_gate_applied),
                @"passed": @(result->face_quality_gate_passed),
                @"engine": @"Apple Vision face capture + feature print + normalized landmarks",
                @"sample_stride": @(params->strict_person_quality ?
                    h3_super_strict_person_quality_stride :
                    (params->person_quality_gate ?
                        h3_super_person_quality_stride :
                        h3_super_face_quality_stride)),
                @"sampled_frames":
                    @(result->face_quality_sampled_frames),
                @"detected_frames":
                    @(result->face_quality_detected_frames),
                @"duplicate_face_frames":
                    @(result->face_quality_duplicate_frames),
                @"detection_rate":
                    @(result->face_quality_detection_rate),
                @"capture_quality_mean":
                    @(result->face_quality_capture_mean),
                @"capture_quality_min":
                    @(result->face_quality_capture_min),
                @"area_fraction_mean":
                    @(result->face_quality_area_mean),
                @"area_fraction_min":
                    @(result->face_quality_area_min),
                @"feature_distance_mean":
                    @(result->face_quality_feature_mean),
                @"feature_distance_max":
                    @(result->face_quality_feature_max),
                @"landmark_distance_mean":
                    @(result->face_quality_landmark_mean),
                @"landmark_distance_max":
                    @(result->face_quality_landmark_max),
                @"mouth_landmark_distance_max":
                    @(result->face_quality_mouth_landmark_max),
                @"contour_landmark_distance_max":
                    @(result->face_quality_contour_landmark_max),
                @"center_jump_mean":
                    @(result->face_quality_center_jump_mean),
                @"center_jump_max":
                    @(result->face_quality_center_jump_max),
                @"log_area_jump_mean":
                    @(result->face_quality_log_area_jump_mean),
                @"log_area_jump_max":
                    @(result->face_quality_log_area_jump_max),
                @"thresholds": @{
                    @"minimum_detection_rate": @(
                        params->strict_person_quality ?
                            h3_super_strict_minimum_detection_rate :
                            (params->person_quality_gate &&
                             params->min_face_area >= 0.006 ?
                                h3_super_person_closeup_minimum_detection_rate :
                                0.80)),
                    @"minimum_landmark_frame_coverage": @0.90,
                    @"minimum_capture_quality": @(
                        params->person_quality_gate ?
                            (params->strict_person_quality ?
                                h3_super_strict_minimum_capture_quality :
                                (params->min_face_area >= 0.006 ?
                                    h3_super_person_closeup_minimum_capture_quality :
                                    h3_super_person_minimum_capture_quality)) :
                            0.0),
                    @"minimum_area_fraction": @(params->min_face_area),
                    @"maximum_feature_distance_mean": @(
                        params->person_quality_gate ?
                            h3_super_person_maximum_feature_mean :
                            h3_super_face_maximum_feature_mean),
                    @"maximum_feature_distance_max": @0.95,
                    @"maximum_landmark_distance_max": @(
                        params->person_quality_gate ?
                            h3_super_person_maximum_landmark :
                            h3_super_face_maximum_landmark),
                    @"maximum_mouth_landmark_distance_max": @(
                        params->person_quality_gate ?
                            h3_super_person_maximum_mouth :
                            h3_super_face_maximum_mouth),
                    @"maximum_contour_landmark_distance_max": @(
                        params->person_quality_gate ?
                            (params->min_face_area >= 0.006 ?
                                h3_super_person_closeup_maximum_contour :
                                h3_super_person_maximum_contour) :
                            h3_super_face_maximum_contour),
                    @"maximum_center_jump": @(
                        params->strict_person_quality ?
                            h3_super_strict_maximum_center_jump : 0.0),
                    @"maximum_log_area_jump": @(
                        params->strict_person_quality ?
                            h3_super_strict_maximum_log_area_jump : 0.0),
                    @"maximum_duplicate_face_frames": @0
                }
            },
            @"body_quality_gate": @{
                @"enabled": @(
                    params->person_quality_gate ||
                    params->output_quality_gate),
                @"applied": @(result->body_quality_gate_applied),
                @"passed": @(result->body_quality_gate_passed),
                @"engine": @"Apple Vision 2D human body pose",
                @"sample_stride": @(params->strict_person_quality ?
                    h3_super_strict_person_quality_stride :
                    (params->person_quality_gate ?
                        h3_super_person_quality_stride :
                        h3_super_face_quality_stride)),
                @"sampled_frames":
                    @(result->body_quality_sampled_frames),
                @"detected_frames":
                    @(result->body_quality_detected_frames),
                @"duplicate_body_frames":
                    @(result->body_quality_duplicate_frames),
                @"detection_rate":
                    @(result->body_quality_detection_rate),
                @"joint_coverage_mean":
                    @(result->body_quality_joint_coverage_mean),
                @"joint_coverage_min":
                    @(result->body_quality_joint_coverage_min),
                @"core_coverage_min":
                    @(result->body_quality_core_coverage_min),
                @"complete_chains_mean":
                    @(result->body_quality_complete_chains_mean),
                @"complete_chains_min":
                    @(result->body_quality_complete_chains_min),
                @"limb_asymmetry_max":
                    @(result->body_quality_limb_asymmetry_max),
                @"thresholds": @{
                    @"minimum_detection_rate": @0.95,
                    @"minimum_joint_coverage_mean": @0.95,
                    @"minimum_joint_coverage_min": @0.85,
                    @"minimum_core_coverage_min": @0.80,
                    @"minimum_complete_chains_mean": @3.50,
                    @"minimum_complete_chains_min": @2.0,
                    @"maximum_limb_asymmetry": @2.25,
                    @"maximum_duplicate_body_frames": @0
                }
            },
            @"candidate_selection": @{
                @"enabled": @(
                    (params->face_quality_gate ||
                     params->person_quality_gate) &&
                    params->quality_attempts > 1),
                @"seed_strategy": @"initial_seed_plus_attempt_index",
                @"maximum_attempts": @(params->quality_attempts),
                @"attempts": @(result->stage1_quality_attempts),
                @"rejected_attempts": @(
                    result->stage1_quality_rejections),
                @"first_seed": @(result->stage1_quality_attempts > 0 ?
                    result->stage1_first_seed : seed),
                @"accepted": @(stage1_admitted),
                @"accepted_seed": stage1_admitted ?
                    @(result->stage1_quality_attempts > 0 ?
                        result->stage1_accepted_seed : seed) : NSNull.null,
                @"rejected_seconds": @(
                    result->stage1_quality_rejected_seconds),
                @"accepted_seconds": @(
                    result->stage1_quality_accepted_seconds),
                @"last_rejection": h3_super_string(
                    result->stage1_quality_last_rejection)
            },
            @"full_candidate_selection": @{
                @"enabled": @(
                    params->person_quality_gate &&
                    params->quality_attempts > 1),
                @"seed_strategy": @"initial_seed_plus_attempt_index",
                @"maximum_attempts": @(params->quality_attempts),
                @"stage1_attempts": @(
                    result->stage1_quality_attempts),
                @"stage1_rejected_attempts": @(
                    result->stage1_quality_rejections),
                @"refine_attempts": @(
                    result->refine_quality_attempts),
                @"output_rejected_attempts": @(
                    result->refine_quality_rejections),
                @"accepted": @(refine_admitted),
                @"accepted_seed": refine_admitted ?
                    @(result->stage1_quality_attempts > 0 ?
                        result->stage1_accepted_seed : seed) : NSNull.null,
                @"refine_rejected_seconds": @(
                    result->refine_quality_rejected_seconds),
                @"refine_accepted_seconds": @(
                    result->refine_quality_accepted_seconds),
                @"refine_total_seconds": @(
                    result->refine_quality_total_seconds),
                @"last_output_rejection": h3_super_string(
                    result->refine_quality_last_rejection)
            },
            @"model_load": @{
                @"attempts": @(result->stage1_model_load_attempts),
                @"seconds": @(result->stage1_model_load_seconds),
                @"includes_post_refine_reloads": @YES
            },
            @"turbo_provenance": turbo_provenance,
            @"path": h3_super_string(stage1_path),
            @"seconds": @(stage1_seconds)
        },
        @"stage2": @{
            @"input_width": @(params->input_width),
            @"input_height": @(params->input_height),
            @"input_frames": @(params->output_frames),
            @"learned_x2_width": @(params->refined_width),
            @"learned_x2_height": @(params->refined_height),
            @"output_width": @(params->output_width),
            @"output_height": @(params->output_height),
            @"output_frames": @(params->output_frames),
            @"fps": @(params->output_fps),
            @"schedule": h3_super_string(
                h3_super_stage2_schedule_name(params->stage2_schedule)),
            @"sigmas": h3_super_string(
                h3_super_stage2_sigmas(params->stage2_schedule)),
            @"sampler": @"euler",
            @"guidance":
                @"positive-only single Transformer call (Sana SimpleDenoiser)",
            @"updates": @(h3_super_stage2_updates(params->stage2_schedule)),
            @"execution_topology": execution_topology,
            @"fused_denoise_decode": @(params->fuse_denoise_decode),
            @"transformer_prefetch": @(params->prefetch_transformer),
            @"transformer_prefetch_completed":
                @(result->transformer_prefetch_completed),
            @"conditioning_prefetch": @(params->prefetch_conditioning),
            @"conditioning_prefetch_completed":
                @(result->conditioning_prefetch_completed),
            @"input_model_prefetch": @(params->prefetch_input_models),
            @"input_model_prefetch_completed":
                @(result->input_model_prefetch_completed),
            @"prefetch_start_step": @(params->prefetch_start_step),
            @"prefetch_started_at_completed_step":
                @(result->prefetch_started_step),
            @"prefetch_stage1_denoise_steps":
                @(result->prefetch_total_steps),
            @"prefetch_start_block": @(params->prefetch_start_block),
            @"prefetch_started_at_completed_block":
                @(result->prefetch_started_block),
            @"prefetch_stage1_dit_blocks":
                @(result->prefetch_total_blocks),
            @"final_evict_blocks": @(params->final_evict_blocks),
            @"transformer": h3_super_string(params->transformer),
            @"refiner": @{
                @"enabled": @(official_stage2),
                @"official_sana_contract":
                    (official_stage2 && default_stage2_schedule) ? @YES : @NO,
                @"official_weights": @(official_stage2),
                @"merge_mode": merged_official_stage2 ?
                    h3_super_uses_hybrid_refiner(params) ?
                        @"selective_bf16_video_self_attention" :
                        @"offline_int8_convrot" :
                    runtime_official_stage2 ?
                        bf16_runtime_stage2 ? @"runtime_bypass_lora_bf16_base" :
                            @"runtime_bypass_lora_int8_base" : @"none",
                @"base_precision": hybrid_official_stage2 ? @"mixed_int8_bf16" :
                    bf16_runtime_stage2 ? @"bf16" :
                    official_stage2 ? @"int8" : @"unknown",
                @"repository": official_stage2 ?
                    @"Lightricks/LTX-2.5" : @"",
                @"revision": official_stage2 ?
                    h3_super_string(h3_super_official_stage2_revision) : @"",
                @"transformer_sha256": official_stage2 ?
                    h3_super_string(official_transformer_sha256) : @"",
                @"transformer_bytes": official_stage2 ?
                    @(official_transformer_bytes) : @0,
                @"base_transformer_sha256": official_stage2 ?
                    h3_super_string(h3_super_official_transformer_sha256) : @"",
                @"base_transformer_bytes": official_stage2 ?
                    @21504034224ULL : @0,
                @"lora": official_stage2 ?
                    h3_super_string(h3_super_official_refiner_lora) : @"",
                @"lora_sha256": official_stage2 ?
                    h3_super_string(h3_super_official_refiner_lora_sha256) : @"",
                @"lora_bytes": official_stage2 ? @8899889568ULL : @0,
                @"strength": @(official_stage2 ? 0.8 : 0.0),
                @"runtime_provenance_gate": merged_official_stage2 ?
                    @"H3SuperVerifyOfficialMergedRefiner verifies the sidecar manifest, source identities, precision mapping, and checkpoint SHA" :
                    runtime_official_stage2 ?
                    @"H3SuperApplyOfficialRefinerLoRA verifies size/SHA and maps all official adapters to bypass low-rank forward hooks without requantizing the INT8 base" :
                    @"not validated"
            },
            @"text_encoder": h3_super_string(params->text_encoder),
            @"video_vae": h3_super_string(params->video_vae),
            @"input_encoder": @"official tiled Video VAE",
            @"highres_first_frame": @{
                @"mode": result->original_first_frame_conditioning ?
                    @"original_asset_sha256_verified" :
                    @"stage1_decoded_fallback",
                @"source_path": params->first_frame ?
                    h3_super_string(params->first_frame) : @"",
                @"sha256": h3_super_string(result->first_frame_sha256),
                @"import_copy_sha256":
                    h3_super_string(result->first_frame_copy_sha256),
                @"bytes": @(result->first_frame_bytes),
                @"import_seconds": @(result->first_frame_import_seconds),
                @"preprocess": params->first_frame ?
                    @"H.264 CRF 18 before bilinear center crop" :
                    @"decoded Stage-1 frame without CRF recompression",
                @"strength": @1.0,
                @"frame_idx": @0
            },
            @"output_person_quality_gate": @{
                @"enabled": @(
                    params->person_quality_gate ||
                    params->output_quality_gate),
                @"applied": @(result->output_quality_gate_applied),
                @"passed": @(result->output_quality_gate_passed),
                @"sample_stride": @(params->strict_person_quality ?
                    h3_super_strict_person_quality_stride :
                    h3_super_person_quality_stride),
                @"seconds": @(result->output_quality_seconds),
                @"face": @{
                    @"sampled_frames": @(
                        result->output_face_sampled_frames),
                    @"detected_frames": @(
                        result->output_face_detected_frames),
                    @"duplicate_frames": @(
                        result->output_face_duplicate_frames),
                    @"detection_rate": @(
                        result->output_face_detection_rate),
                    @"capture_quality_mean": @(
                        result->output_face_capture_mean),
                    @"capture_quality_min": @(
                        result->output_face_capture_min),
                    @"area_fraction_mean": @(
                        result->output_face_area_mean),
                    @"area_fraction_min": @(
                        result->output_face_area_min),
                    @"feature_distance_mean": @(
                        result->output_face_feature_mean),
                    @"feature_distance_max": @(
                        result->output_face_feature_max),
                    @"landmark_distance_max": @(
                        result->output_face_landmark_max),
                    @"mouth_landmark_distance_max": @(
                        result->output_face_mouth_landmark_max),
                    @"contour_landmark_distance_max": @(
                        result->output_face_contour_landmark_max),
                    @"center_jump_mean": @(
                        result->output_face_center_jump_mean),
                    @"center_jump_max": @(
                        result->output_face_center_jump_max),
                    @"log_area_jump_mean": @(
                        result->output_face_log_area_jump_mean),
                    @"log_area_jump_max": @(
                        result->output_face_log_area_jump_max),
                    @"thresholds": @{
                        @"minimum_detection_rate": @(
                            params->strict_person_quality ?
                                h3_super_strict_minimum_detection_rate :
                                (params->min_face_area >= 0.006 ?
                                    h3_super_person_closeup_minimum_detection_rate :
                                    0.85)),
                        @"minimum_landmark_frame_coverage": @0.90,
                        @"minimum_capture_quality": @(
                            params->strict_person_quality ?
                                h3_super_strict_minimum_capture_quality :
                                (params->min_face_area >= 0.006 ?
                                    h3_super_person_closeup_minimum_capture_quality :
                                    h3_super_person_minimum_capture_quality)),
                        @"minimum_area_fraction": @(params->min_face_area),
                        @"maximum_feature_distance_mean": @0.65,
                        @"maximum_feature_distance_max": @0.95,
                        @"maximum_landmark_distance_max": @0.60,
                        @"maximum_mouth_landmark_distance_max": @0.60,
                        @"maximum_contour_landmark_distance_max": @(
                            params->min_face_area >= 0.006 ?
                                h3_super_output_closeup_maximum_contour :
                                0.70),
                        @"maximum_center_jump": @(
                            params->strict_person_quality ?
                                h3_super_strict_maximum_center_jump : 0.0),
                        @"maximum_log_area_jump": @(
                            params->strict_person_quality ?
                                h3_super_strict_maximum_log_area_jump : 0.0),
                        @"maximum_duplicate_frames": @0
                    }
                },
                @"body": @{
                    @"applied": @(result->output_body_gate_applied),
                    @"passed": @(result->output_body_gate_passed),
                    @"sampled_frames": @(
                        result->output_body_sampled_frames),
                    @"detected_frames": @(
                        result->output_body_detected_frames),
                    @"duplicate_frames": @(
                        result->output_body_duplicate_frames),
                    @"detection_rate": @(
                        result->output_body_detection_rate),
                    @"joint_coverage_mean": @(
                        result->output_body_joint_coverage_mean),
                    @"joint_coverage_min": @(
                        result->output_body_joint_coverage_min),
                    @"core_coverage_min": @(
                        result->output_body_core_coverage_min),
                    @"complete_chains_mean": @(
                        result->output_body_complete_chains_mean),
                    @"complete_chains_min": @(
                        result->output_body_complete_chains_min),
                    @"limb_asymmetry_max": @(
                        result->output_body_limb_asymmetry_max)
                }
            },
            @"audio_vae": h3_super_string(params->audio_vae),
            @"latent_upscaler": h3_super_string(params->latent_upscaler),
            @"attention": params->sol_attention ?
                @"layer 0 dense; layers 1-47 Metal Sol-Attn" :
                @"dense ComfyUI implementation",
            @"sol_attention": @(params->sol_attention),
            @"sol_taus": params->sol_attention ?
                (params->stage2_schedule ==
                    H3_SUPER_STAGE2_SCHEDULE_DEFAULT ?
                    @[@1.0, @1.25, @1.5] :
                 params->stage2_schedule ==
                    H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9 ?
                    @[@1.0, @1.25] : @[@1.0]) : @[],
            @"sol_threshold": params->sol_attention ? @"diag" : @"",
            @"sol_threshold_backend": params->sol_attention ?
                @"fused-metal" : @"",
            @"sol_query_block_size": params->sol_attention ? @64 : @0,
            @"decoder": decoder,
            @"delivery_codec": @"H.264",
            @"delivery_crf": @18,
            @"delivery_bit_depth": @8,
            @"taehv_checkpoint": params->decoder == H3_SUPER_DECODER_TAEHV ?
                h3_super_string(params->taehv_checkpoint) : @"",
            @"taehv_compute_dtype":
                params->decoder == H3_SUPER_DECODER_TAEHV ?
                h3_super_string(params->taehv_compute_dtype) : @"",
            @"taehv_temporal_mode":
                params->decoder == H3_SUPER_DECODER_TAEHV ?
                h3_super_string(params->taehv_temporal_mode) : @"",
            @"taehv_keep_on_device":
                @(params->decoder == H3_SUPER_DECODER_TAEHV &&
                  params->taehv_keep_on_device),
            @"seconds": @(result->stage2_seconds),
            @"total_attempt_seconds": @(
                result->refine_quality_total_seconds),
            @"queue_and_execution_seconds": @(result->queue_seconds)
        },
        @"timing": @{
            @"preflight_seconds": @(result->preflight_seconds),
            @"stage1_model_load_seconds": @(
                result->stage1_model_load_seconds),
            @"stage1_seconds": @(stage1_seconds),
            @"handoff_seconds": @(result->handoff_seconds),
            @"first_frame_import_seconds":
                @(result->first_frame_import_seconds),
            @"stage2_seconds": @(result->stage2_seconds),
            @"stage2_total_attempt_seconds": @(
                result->refine_quality_total_seconds),
            @"conditioning_seconds": @(result->conditioning_seconds),
            @"input_prepare_seconds": @(result->input_prepare_seconds),
            @"denoise_seconds": @(result->denoise_seconds),
            @"decode_seconds": @(result->decode_seconds),
            @"denoise_decode_seconds":
                @(result->denoise_decode_seconds),
            @"output_copy_seconds": @(result->output_copy_seconds),
            @"output_quality_seconds": @(
                result->output_quality_seconds),
            @"output_quality_total_seconds": @(
                result->output_quality_total_seconds),
            @"backend_release_seconds": @(result->backend_release_seconds),
            @"transformer_prefetch_seconds":
                @(result->transformer_prefetch_seconds),
            @"transformer_prefetch_wait_seconds":
                @(result->transformer_prefetch_wait_seconds),
            @"conditioning_prefetch_seconds":
                @(result->conditioning_prefetch_seconds),
            @"conditioning_prefetch_wait_seconds":
                @(result->conditioning_prefetch_wait_seconds),
            @"input_model_prefetch_seconds":
                @(result->input_model_prefetch_seconds),
            @"total_seconds": @(total_seconds),
            @"total_scope": @"post-CLI validation through final admission; includes preflight and all H3 model loads"
        },
        @"memory": @{
            @"scope": @"system-wide values sampled through ComfyUI; not process RSS",
            @"policy": params->allow_swap ?
                @"speed-first; swap explicitly accepted and recorded" :
                @"speed-first; swapout is warning-only and E2E stability is the admission criterion",
            @"ram_total_bytes": @(result->backend_ram_total),
            @"ram_free_before_bytes": @(result->backend_ram_free_before),
            @"ram_free_min_sampled_bytes": @(result->backend_ram_free_min),
            @"ram_used_peak_sampled_bytes": @(
                result->backend_ram_total >= result->backend_ram_free_min ?
                result->backend_ram_total - result->backend_ram_free_min : 0),
            @"ram_free_after_release_bytes": @(
                result->backend_ram_free_after_release),
            @"system_swapouts_before": @(result->system_swapouts_before),
            @"system_swapouts_after": @(result->system_swapouts_after),
            @"system_swapouts_delta": @(
                result->system_swapouts_after >= result->system_swapouts_before ?
                result->system_swapouts_after - result->system_swapouts_before : 0),
            @"swapouts_available": @(result->swapouts_available),
            @"no_swap_gate_passed": @(result->no_swap_gate_passed),
            @"allow_swap": @(params->allow_swap)
        },
        @"output": @{
            @"path": h3_super_string(output_path),
            @"backend_path": h3_super_string(result->backend_output),
            @"bytes": @(result->output_bytes)
        },
        @"known_differences_from_sana_v2": differences
    };
    NSData *data = h3_super_json_data(payload, error, error_size);
    if (!data) return 0;
    NSString *path = h3_super_string(params->telemetry_path);
    NSString *parent = path.stringByDeletingLastPathComponent;
    NSError *file_error = nil;
    if (parent.length && ![NSFileManager.defaultManager
            createDirectoryAtPath:parent withIntermediateDirectories:YES
                       attributes:nil error:&file_error]) {
        return h3_super_fail(error, error_size,
                             "cannot create telemetry directory: %s",
                             file_error.localizedDescription.UTF8String);
    }
    NSString *temporary = [path stringByAppendingFormat:@".partial-%d", getpid()];
    if (![data writeToFile:temporary options:NSDataWritingAtomic error:&file_error]) {
        return h3_super_fail(error, error_size, "cannot write telemetry: %s",
                             file_error.localizedDescription.UTF8String);
    }
    [NSFileManager.defaultManager removeItemAtPath:path error:nil];
    if (![NSFileManager.defaultManager moveItemAtPath:temporary toPath:path
                                                error:&file_error]) {
        [NSFileManager.defaultManager removeItemAtPath:temporary error:nil];
        return h3_super_fail(error, error_size, "cannot install telemetry: %s",
                             file_error.localizedDescription.UTF8String);
    }
    return 1;
}
