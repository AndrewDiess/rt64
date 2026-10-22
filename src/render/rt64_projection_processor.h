//
// RT64
//

#pragma once

#include "hle/rt64_game_frame.h"
#include "../include/rt64_extended_gbi.h"

#include "rt64_buffer_uploader.h"

namespace RT64 {
    // Viewport origin (beyond the G_EX_ORIGIN_* values) for full-screen overlays drawn in 3D, e.g. Rogue Squadron's
    // pause darkening and fades: never widened, drawn stretched across the wide target.
    constexpr uint16_t ViewportOriginStretch = G_EX_ORIGIN_NONE + 1;

    // Whether a 3D viewport is widened with the Expand aspect ratio (its projection adjusted and drawn across the
    // wide target). The projection processor and the framebuffer renderer must agree, or the scene is stretched.
    // pixelToDisplay corrects for non-square framebuffer pixels (Rogue Squadron's 400x448 intro framebuffer is
    // shown at about 4:3), and one pixel (4 in 10.2) of slack allows the common width - 1 scissor.
    inline bool usesWideViewport(const FixedRect &intersectionRect, const FixedRect &fbScissorRect, uint16_t viewportOrigin, float pixelToDisplay) {
        if (intersectionRect.isEmpty()) {
            return false;
        }

        const int32_t OnePixel = 4;
        const bool coversWholeWidth = (intersectionRect.ulx <= fbScissorRect.ulx + OnePixel) && (intersectionRect.lrx + OnePixel >= fbScissorRect.lrx);
        const bool horizontalRatio = (float(intersectionRect.width(true, true)) * pixelToDisplay) > float(intersectionRect.height(true, true));
        return (viewportOrigin == G_EX_ORIGIN_NONE) && coversWholeWidth && horizontalRatio;
    }

    struct ProjectionProcessor {
        std::unique_ptr<BufferUploader> bufferUploader;
        std::vector<BufferUploader::Upload> uploads;

        struct ProcessParams {
            RenderWorker *worker = nullptr;
            WorkloadQueue *workloadQueue = nullptr;
            GameFrame *curFrame = nullptr;
            const GameFrame *prevFrame = nullptr;
            float curFrameWeight = 1.0f;
            float prevFrameWeight = 0.0f;
            float aspectRatioScale = 1.0f;
            float pixelToDisplay = 1.0f; // Displayed width of a framebuffer pixel relative to its height.
        };

        ProjectionProcessor();
        ~ProjectionProcessor();
        void setup(RenderWorker *worker);
        void process(const ProcessParams &p);
        void processScene(const ProcessParams &p, const GameScene &scene, size_t sceneIndex);
        void upload(const ProcessParams &p);
    };
};