#pragma once
#import <Foundation/Foundation.h>

namespace tc {
// Pure metadata/composition: no model session, filesystem access or plan.
NSDictionary *image_workflows_catalog();
NSDictionary *image_workflow_request(NSDictionary *input);
}
