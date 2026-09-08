#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

#include <algorithm>

namespace {

VNFeaturePrintObservation *feature_print(NSString *path, NSError **error) {
    NSURL *url = [NSURL fileURLWithPath:path];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!source) {
        if (error)
            *error = [NSError errorWithDomain:@"TurboCiderVisionQuality"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    [@"cannot open image: "
                                                        stringByAppendingString:path]}];
        return nil;
    }
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image) {
        if (error)
            *error = [NSError errorWithDomain:@"TurboCiderVisionQuality"
                                         code:2
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    [@"cannot decode image: "
                                                        stringByAppendingString:path]}];
        return nil;
    }
    VNGenerateImageFeaturePrintRequest *request =
        [[VNGenerateImageFeaturePrintRequest alloc] init];
    request.revision = VNGenerateImageFeaturePrintRequestRevision2;
    request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
    VNImageRequestHandler *handler =
        [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
    BOOL completed = [handler performRequests:@[request] error:error];
    CGImageRelease(image);
    if (!completed)
        return nil;
    id observation = request.results.firstObject;
    if (![observation isKindOfClass:[VNFeaturePrintObservation class]]) {
        if (error)
            *error = [NSError errorWithDomain:@"TurboCiderVisionQuality"
                                         code:3
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    @"Vision returned no feature print"}];
        return nil;
    }
    return (VNFeaturePrintObservation *)observation;
}

int fail(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    return 1;
}

} // namespace

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3 || ((argc - 1) % 2) != 0)
            return fail(@"usage: vision-feature-distance REFERENCE CANDIDATE "
                        "[REFERENCE CANDIDATE ...]");

        NSMutableArray *pairs = [NSMutableArray array];
        double total = 0.0;
        float maximum = 0.0f;
        for (int index = 1; index < argc; index += 2) {
            NSString *reference = [NSString stringWithUTF8String:argv[index]];
            NSString *candidate = [NSString stringWithUTF8String:argv[index + 1]];
            NSError *error = nil;
            VNFeaturePrintObservation *left = feature_print(reference, &error);
            if (!left)
                return fail(error.localizedDescription ?: @"Vision reference request failed");
            VNFeaturePrintObservation *right = feature_print(candidate, &error);
            if (!right)
                return fail(error.localizedDescription ?: @"Vision candidate request failed");
            float distance = 0.0f;
            if (![left computeDistance:&distance
                      toFeaturePrintObservation:right
                                         error:&error])
                return fail(error.localizedDescription ?: @"Vision distance failed");
            total += distance;
            maximum = std::max(maximum, distance);
            [pairs addObject:@{
                @"reference" : reference,
                @"candidate" : candidate,
                @"distance" : @(distance),
            }];
        }
        NSDictionary *result = @{
            @"schema_version" : @1,
            @"metric" : @"VNFeaturePrintObservation distance",
            @"vision_request_revision" :
                @(VNGenerateImageFeaturePrintRequestRevision2),
            @"image_crop_and_scale" : @"scale_fill",
            @"lower_is_more_similar" : @YES,
            @"operating_system" : NSProcessInfo.processInfo.operatingSystemVersionString,
            @"pair_count" : @(pairs.count),
            @"mean_distance" : @(total / pairs.count),
            @"maximum_distance" : @(maximum),
            @"pairs" : pairs,
        };
        NSError *error = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:result
                                                       options:NSJSONWritingPrettyPrinted |
                                                               NSJSONWritingSortedKeys
                                                         error:&error];
        if (!data)
            return fail(error.localizedDescription ?: @"cannot encode result JSON");
        fwrite(data.bytes, 1, data.length, stdout);
        fputc('\n', stdout);
        return 0;
    }
}
