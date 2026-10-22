//
// RT64
//
// Copyright (c) 2026 RT64 Contributors
// SPDX-License-Identifier: MIT
//

#include "rt64_gbi_f5rogue.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "hle/rt64_interpreter.h"
#include "render/rt64_projection_processor.h"

#include "rt64_gbi_rdp.h"

// The microcode is emulated at the vertex level: vertices are transformed on the CPU with the same fixed point
// steps as the RSP (for clip flags, culling, fog, billboards and the terrain depth sort), and RT64 draws the
// triangles from the model space positions and matrices like it does for F3D. Clipping is left to the GPU: the
// microcode clips against near/far and against x/y planes at twice the screen (guard band), so the visible
// result is the same as drawing the unclipped triangle with near/far clipping and the scissor.
//
// RT64 assumes F3D conventions in a few places. The mappings are:
// - Screen y: the microcode does no y negation (screen = ndc * scale + translate), RT64 negates y like F3D. The
//   projection given to RT64 has its y column negated, so the viewport can be passed as is.
// - Geometry mode: only bits that mean the same as in F3D are given to RT64 (z-buffer, texture, shade, smooth
//   shading). Culling (bit 13, clockwise on screen) is done here, bit 12 is "no texture perspective" instead of
//   F3D's front culling, and fog (bit 16) is computed here and stored in the vertex alpha, like the microcode.
// - Matrices: there is no matrix stack. Loading a modelview computes modelview x projection, loading a projection
//   doesn't update the combined matrix, so RT64 is given the previous combined matrix in that case.
// - Vertex colors and texture coordinates are set per triangle, so RT64's vertex copy is changed (copied if it
//   was already drawn) before each triangle when they differ.
//
// Widescreen handling specific to Rogue Squadron (not microcode behavior, see "Widescreen" below): HUD pieces are
// anchored to the screen edges, full-screen overlays stretch, the menu wipe's sub-pixel gaps are closed and the
// game's one-pixel-short scissors are kept at full size.

// Debug aid (see debugDump below), set by the host.
uint32_t f5rogueDebugDumpTask = 0;

namespace RT64 {
    namespace GBI_F5ROGUE {
        // DMEM layout.
        constexpr uint32_t DmemSize = 0x1000;
        constexpr uint32_t DmemMask = DmemSize - 1;
        constexpr uint32_t DmemTaskDataSize = 0x800;
        constexpr uint32_t DmemClipRatio = 0x006;
        constexpr uint32_t DmemScreenMinXY = 0x062;
        constexpr uint32_t DmemScreenMinZ = 0x066;
        constexpr uint32_t DmemMoveMemTable = 0x046;
        constexpr uint32_t DmemSortKeyScale = 0x074;
        constexpr uint32_t DmemMoveWordTable = 0x0C6;
        constexpr uint32_t DmemOtherModeH = 0x10C;
        constexpr uint32_t DmemOtherModeL = 0x110;
        constexpr uint32_t DmemViewportScale = 0x120;
        constexpr uint32_t DmemViewportTranslate = 0x128;
        constexpr uint32_t DmemGeometryMode = 0x134;
        constexpr uint32_t DmemTexture = 0x138;
        constexpr uint32_t DmemPerspNorm = 0x13E;
        constexpr uint32_t DmemTexcoordScaleInteger = 0x140;
        constexpr uint32_t DmemTexcoordScaleFraction = 0x148;
        constexpr uint32_t DmemFogMultiplier = 0x160;
        constexpr uint32_t DmemFogOffset = 0x164;
        constexpr uint32_t DmemFogScale = 0x168;
        constexpr uint32_t DmemVertexInput = 0x280;
        constexpr uint32_t DmemGridColorMap = 0x380;
        constexpr uint32_t DmemGridHeightMap = 0x4E0;
        constexpr uint32_t DmemProjection = 0x590;
        constexpr uint32_t DmemModelView = 0x5D0;
        constexpr uint32_t DmemCombined = 0x610;
        constexpr uint32_t DmemColorTable = 0xB70;
        constexpr uint32_t DmemSortHead = 0xC70;
        constexpr uint32_t DmemGridHeights = 0xCB4;

        // Matrices are in the N64 layout: 16 integer halfwords followed by 16 fraction halfwords.
        constexpr uint32_t MatrixFractionOffset = 0x20;

        // Display list blocks: an 8-byte header (next block, previous block) and 32 commands. The microcode copies
        // 0x110 bytes to DMEM 0x170 and starts on the next block when the read pointer is exactly 0x278.
        constexpr uint32_t BlockHeaderSize = 0x8;
        constexpr uint32_t BlockCommandsEnd = 0x108;

        // Vertex buffer.
        constexpr uint32_t VertexStride = 0x28;
        constexpr uint32_t VertexCount = 32;

        // Geometry mode bits read by the microcode.
        constexpr uint32_t GeometryZBuffer = 0x00000001;
        constexpr uint32_t GeometryTexture = 0x00000002;
        constexpr uint32_t GeometryShade = 0x00000004;
        constexpr uint32_t GeometrySmooth = 0x00000200;
        constexpr uint32_t GeometryNoPerspective = 0x00001000;
        constexpr uint32_t GeometryCullClockwise = 0x00002000;
        constexpr uint32_t GeometryFog = 0x00010000;
        constexpr uint32_t GeometryPassedToRT64 = GeometryZBuffer | GeometryTexture | GeometryShade | GeometrySmooth;

        // Clip flags: frustum planes (against w) in bits 4-6 (below) and 12-14 (above), guard band planes (against
        // w * clip ratio) in bits 0-2 and 8-10.
        constexpr uint16_t ClipRejectMask = 0x7070;
        constexpr uint16_t ClipRequiredMask = 0x4343;

        // TRI_GEN.
        constexpr uint32_t TriGenSize = 40;
        constexpr uint32_t TriGenMaxSide = 5;
        constexpr uint8_t TriGenSecondTriangle = 0x40;
        constexpr uint8_t TriGenFiller = 0x80;

        // Same depth range as the RSP's viewport conversion.
        constexpr float DepthRange = 1024.0f;

        struct Vertex {
            // Model space position (the VTX input), used by RT64 to transform the vertex again on the GPU.
            int16_t position[3];

            // Vertex buffer record in the microcode.
            int32_t clip[4];
            uint32_t rgba;
            int16_t s;
            int16_t t;
            int16_t screen[3];
            uint16_t screenZFraction;
            int32_t invW;
            uint16_t clipFlags;
        };

        struct LoadedAttributes {
            uint32_t rgba;
            float s;
            float t;
        };

        struct CallStackEntry {
            uint32_t blockAddress;
            uint32_t readOffset;
        };

        // A HUD piece anchored to a screen edge (10.2 coordinates).
        struct HudRegion {
            int32_t ulx, uly, lrx, lry;
            uint16_t origin;
        };

        struct Context {
            std::array<uint8_t, DmemSize> dmem = {};
            std::array<Vertex, VertexCount> vertices = {};

            // Color and texture coordinates of RT64's copy of each vertex.
            std::array<LoadedAttributes, VertexCount> loaded = {};

            std::vector<CallStackEntry> callStack;
            uint32_t blockAddress = 0;
            uint32_t nextBlockAddress = 0;
            DisplayList *blockEnd = nullptr;
            bool blockFetched = false;
            std::array<int16_t, 7> syncedViewport = {};
            uint16_t syncedViewportAnchor = G_EX_ORIGIN_NONE;
            bool viewportSynced = false;
            std::array<GBIFunction, 0x40> rdpFunctions = {};

            // Widescreen: the edge the viewport is anchored to for HUD triangles, the last vertex load (reloaded
            // when the anchor changes, since RT64 binds the viewport at vertex load), how far each loaded vertex was
            // moved on screen, the HUD pieces anchored in this and the previous task, and the last scissor.
            uint16_t viewportAnchor = G_EX_ORIGIN_NONE;
            uint32_t lastLoadAddress = 0;
            uint32_t lastLoadCount = 0;
            std::array<float, VertexCount> screenShiftX = {};
            std::vector<HudRegion> hudRegions;
            std::vector<HudRegion> hudRegionsPrevious;
            std::array<int32_t, 4> previousScissor = { -1, -1, -1, -1 };
        };

        // RT64 only runs one state at a time, so the microcode state lives here like a DMEM would.
        static Context context;

        // Arithmetic helpers that follow the RSP's vector unit.

        static int32_t clamp32(int64_t value) {
            return int32_t(std::clamp<int64_t>(value, INT32_MIN, INT32_MAX));
        }

        static int16_t clamp16(int64_t value) {
            return int16_t(std::clamp<int64_t>(value, INT16_MIN, INT16_MAX));
        }

        // Product of two s15.16 values (or s15.16 and an unsigned 0.16 value), floored and saturated.
        static int32_t mulFixed(int32_t a, int64_t b) {
            return clamp32((int64_t(a) * b) >> 16);
        }

        // 1/x for an s15.16 value, as an s15.16 value. The microcode uses VRCPH/VRCPL and one Newton-Raphson step,
        // which is within a few units of the exact result.
        static int32_t reciprocal(int32_t value) {
            if (value == 0) {
                return INT32_MAX;
            }

            return clamp32((int64_t(1) << 32) / value);
        }

        // VCH/VCL clip test on 32-bit values: "below" is value <= -limit and "above" is value >= limit.
        static void clipTest(int32_t value, int32_t limit, bool &below, bool &above) {
            if ((value ^ limit) < 0) {
                above = (limit < 0);
                below = (int64_t(value) + limit) <= 0;
            }
            else {
                below = (limit < 0);
                above = (int64_t(value) - limit) >= 0;
            }
        }

        // DMEM access (big endian, addresses wrap like the RSP's).

        static uint8_t dmemRead8(uint32_t address) {
            return context.dmem[address & DmemMask];
        }

        static uint16_t dmemRead16(uint32_t address) {
            return uint16_t((dmemRead8(address) << 8) | dmemRead8(address + 1));
        }

        static int16_t dmemReadS16(uint32_t address) {
            return int16_t(dmemRead16(address));
        }

        static uint32_t dmemRead32(uint32_t address) {
            return (uint32_t(dmemRead16(address)) << 16) | dmemRead16(address + 2);
        }

        static void dmemWrite8(uint32_t address, uint8_t value) {
            context.dmem[address & DmemMask] = value;
        }

        static void dmemWrite16(uint32_t address, uint16_t value) {
            dmemWrite8(address, uint8_t(value >> 8));
            dmemWrite8(address + 1, uint8_t(value));
        }

        static void dmemWrite32(uint32_t address, uint32_t value) {
            dmemWrite16(address, uint16_t(value >> 16));
            dmemWrite16(address + 2, uint16_t(value));
        }

        // s15.16 value from an integer halfword and a fraction halfword.
        static int32_t dmemReadFixed(uint32_t integerAddress, uint32_t fractionAddress) {
            return int32_t((uint32_t(dmemRead16(integerAddress)) << 16) | dmemRead16(fractionAddress));
        }

        static int32_t dmemMatrixElement(uint32_t matrixAddress, uint32_t row, uint32_t column) {
            const uint32_t offset = (row * 4 + column) * 2;
            return dmemReadFixed(matrixAddress + offset, matrixAddress + MatrixFractionOffset + offset);
        }

        static hlslpp::float4x4 dmemMatrix(uint32_t matrixAddress) {
            float m[16];
            for (uint32_t i = 0; i < 16; i++) {
                m[i] = float(dmemMatrixElement(matrixAddress, i / 4, i % 4)) / 65536.0f;
            }

            return hlslpp::float4x4(
                m[0], m[1], m[2], m[3],
                m[4], m[5], m[6], m[7],
                m[8], m[9], m[10], m[11],
                m[12], m[13], m[14], m[15]
            );
        }

        // Negates the y column of a matrix. RT64 negates y in the screen transform like F3D and the microcode doesn't.
        static hlslpp::float4x4 negateY(const hlslpp::float4x4 &m) {
            const hlslpp::float4x4 flip(
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, -1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            );

            return hlslpp::mul(m, flip);
        }

        static uint8_t rdramRead8(State *state, uint32_t address) {
            return state->RDRAM[address ^ 3];
        }

        // RSP DMA from RDRAM to DMEM. The length is the SP_RD_LEN value: bits 0-11 length - 1 (rounded up to 8
        // bytes), bits 12-19 row count - 1, bits 20-31 row skip. Addresses are aligned down to 8 bytes.
        static void dmaRead(State *state, uint32_t dmemAddress, uint32_t rdramAddress, uint32_t lengthRegister) {
            const uint32_t rowLength = (lengthRegister & 0xFF8) + 8;
            const uint32_t rowCount = ((lengthRegister >> 12) & 0xFF) + 1;
            const uint32_t rowSkip = (lengthRegister >> 20) & 0xFF8;
            uint32_t dst = dmemAddress & 0xFF8;
            uint32_t src = rdramAddress & RSP_DMA_MASK;
            for (uint32_t r = 0; r < rowCount; r++) {
                for (uint32_t i = 0; i < rowLength; i++) {
                    // The rest would go to IMEM.
                    if (dst >= DmemSize) {
                        return;
                    }

                    context.dmem[dst++] = rdramRead8(state, src++);
                }

                src += rowSkip;
            }
        }

        // Debug aid for comparing with the reference renderer (tools5ref in the Rogue Squadron project): while
        // f5rogueDebugDumpTask is nonzero (set by the host for a saved task), every triangle and texture rectangle
        // drawn is written to C:/projects/swrs-gfxsnap/task_<n>.hle.tsv in the format of tools5rdp.py: per vertex
        // x y (pixels) z (RDP z) r g b a s t w, sorted by y, plus a "clipped" column for triangles the microcode
        // would clip (it then draws several smaller triangles).
        static FILE *debugDumpFile = nullptr;
        static uint32_t debugDumpOpenTask = 0;
        static uint32_t debugDumpIndex = 0;

        static FILE *debugDump() {
            if (f5rogueDebugDumpTask != debugDumpOpenTask) {
                if (debugDumpFile != nullptr) {
                    fclose(debugDumpFile);
                    debugDumpFile = nullptr;
                }

                debugDumpOpenTask = f5rogueDebugDumpTask;
                debugDumpIndex = 0;
                if (debugDumpOpenTask != 0) {
                    char path[128];
                    snprintf(path, sizeof(path), "C:/projects/swrs-gfxsnap/task_%u.hle.tsv", debugDumpOpenTask);
                    debugDumpFile = fopen(path, "w");
                }
            }

            return debugDumpFile;
        }

        static void debugDumpRect(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls, int16_t ult, int16_t dsdx, int16_t dtdy, bool flip) {
            FILE *f = debugDump();
            if (f == nullptr) {
                return;
            }

            fprintf(f, "rect\t%u\t%02X\t%u\t%g %g %g %g %d %d %d %d\n", debugDumpIndex++, flip ? 0xE5 : 0xE4, tile,
                ulx / 4.0, uly / 4.0, lrx / 4.0, lry / 4.0, uls, ult, dsdx, dtdy);
        }

        static void debugDumpTriangle(State *state, uint32_t a, uint32_t b, uint32_t c, uint32_t flatIndex, uint32_t geometryMode, bool clipped);

        // Widescreen helpers (below).
        static uint16_t hudEdgeOrigin(int32_t left, int32_t right, int32_t fbWidth);
        static void adjustScissor(State *state, uint32_t &w0, uint32_t &w1);

        static void sendRDP(State *state, uint32_t w0, uint32_t w1) {
            const uint32_t opCode = (w0 >> 24) & 0x3F;
            if (opCode == (G_RDPSETOTHERMODE & 0x3F)) {
                // Keep the RSP's copy of the other mode in sync too.
                state->rsp->setOtherMode(w0 & 0xFFFFFF, w1);
                return;
            }

            if (opCode == (G_SETSCISSOR & 0x3F)) {
                adjustScissor(state, w0, w1);
            }

            const GBIFunction function = context.rdpFunctions[opCode];
            if (function != nullptr) {
                DisplayList command;
                command.w0 = w0;
                command.w1 = w1;
                DisplayList *commandPointer = &command;
                function(state, &commandPointer);
            }
            else {
                RT64_LOG_PRINTF("F5Rogue: unsupported RDP command 0x%08X 0x%08X", w0, w1);
            }
        }

        // RDP texture rectangle in its native 128-bit format.
        // widescreenRules: HUD anchoring and overlay stretching apply (RDP texture rectangles from the display list,
        // not TEXRECT_GEN billboards).
        static void drawTexRect(State *state, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, bool widescreenRules) {
            const int32_t ulx = (w1 >> 12) & 0xFFF;
            const int32_t uly = w1 & 0xFFF;
            const int32_t lrx = (w0 >> 12) & 0xFFF;
            const int32_t lry = w0 & 0xFFF;
            const uint8_t tile = (w1 >> 24) & 0x7;
            const int16_t uls = int16_t(w2 >> 16);
            const int16_t ult = int16_t(w2);
            const int16_t dsdx = int16_t(w3 >> 16);
            const int16_t dtdy = int16_t(w3);
            const bool flip = ((w0 >> 24) & 0x3F) == (G_TEXRECTFLIP & 0x3F);
            debugDumpRect(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, flip);
            if (!widescreenRules) {
                state->rdp->drawTexRect(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, flip);
                return;
            }

            // Widescreen. Rectangles that lie entirely in the outer 30% of the screen (the lives counter, wingman icon,
            // radar) are anchored to that edge, so the HUD placement setting can move them into the extended area.
            // Everything else stays centered (messages, the title image). Billboards use TEXRECT_GEN and are unaffected.
            // Anchors only matter when the HUD placement allows it; 4:3 screens ignore them.
            const int32_t fbWidth = int32_t(state->rdp->colorImage.width) * 4;
            ExtendedAlignment alignment = state->rdp->extended.global.rect;
            int32_t ax0 = ulx, ax1 = lrx;

            // Screen overlays: a rectangle covering the scissor's full height and reaching its left or right edge is a
            // transition, not HUD (the wipe when a mission ends is one rectangle growing from the right edge). It
            // stretches across the wide screen so it covers the extended area and finishes at the same time.
            const FixedRect &scissor = state->rdp->scissorRectStack[state->rdp->scissorStackSize - 1];
            const bool screenOverlay = (uly <= scissor.uly) && (lry >= scissor.lry) && ((ulx <= scissor.ulx) || (lrx >= scissor.lrx)) &&
                (alignment.leftOrigin == G_EX_ORIGIN_NONE) && (alignment.rightOrigin == G_EX_ORIGIN_NONE);
            if (screenOverlay) {
                // Edges within a pixel of the scissor edge snap to it: the wipe's last frame starts at x = 0.25, a sliver
                // that stretching would make visible.
                const int32_t ox0 = ((ulx > scissor.ulx) && (ulx - scissor.ulx <= 4)) ? scissor.ulx : ulx;
                const int32_t ox1 = ((lrx < scissor.lrx) && (scissor.lrx - lrx <= 4)) ? scissor.lrx : lrx;
                const uint8_t previousAspect = state->rdp->extended.global.rectAspect;
                state->rdp->extended.global.rectAspect = G_EX_ASPECT_STRETCH;
                state->rdp->drawTexRect(ox0, uly, ox1, lry, tile, uls, ult, dsdx, dtdy, flip, alignment);
                state->rdp->extended.global.rectAspect = previousAspect;
                return;
            }

            if ((alignment.leftOrigin == G_EX_ORIGIN_NONE) && (alignment.rightOrigin == G_EX_ORIGIN_NONE)) {
                // Larger pieces (24 x 24 pixels or more: the lives icon, wingman icon, radar strips) in an outer band
                // anchor to that edge. Smaller ones (digits, labels) follow a larger anchored piece they lie inside, from
                // this or the previous task, with a margin of 8 pixels (the radar ring is four horizontal strips and a
                // small marker straddles the line between two of them). Text stays centered.
                const int32_t MinAnchorSize = 24 * 4;
                const int32_t Margin = 8 * 4;
                uint16_t origin = G_EX_ORIGIN_NONE;
                if (((lrx - ulx) >= MinAnchorSize) && ((lry - uly) >= MinAnchorSize)) {
                    origin = hudEdgeOrigin(ulx, lrx, fbWidth);
                    if (origin != G_EX_ORIGIN_NONE) {
                        context.hudRegions.push_back({ ulx, uly, lrx, lry, origin });
                    }
                }
                else {
                    for (const std::vector<HudRegion> *regions : { &context.hudRegions, &context.hudRegionsPrevious }) {
                        for (const HudRegion &region : *regions) {
                            if ((ulx >= region.ulx - Margin) && (lrx <= region.lrx + Margin) && (uly >= region.uly - Margin) && (lry <= region.lry + Margin)) {
                                origin = region.origin;
                                break;
                            }
                        }

                        if (origin != G_EX_ORIGIN_NONE) {
                            break;
                        }
                    }
                }

                if (origin != G_EX_ORIGIN_NONE) {
                    alignment.leftOrigin = alignment.rightOrigin = origin;
                    if (origin == G_EX_ORIGIN_RIGHT) {
                        // Coordinates are relative to the origin: the right edge.
                        ax0 -= fbWidth;
                        ax1 -= fbWidth;
                    }
                }
            }

            state->rdp->drawTexRect(ax0, uly, ax1, lry, tile, uls, ult, dsdx, dtdy, flip, alignment);
        }

        // Widescreen: the game sets a full-frame scissor and then the same scissor one pixel smaller on the right and
        // bottom (e.g. (0,0)-(400,448) then (0,0)-(399,447)), leaving the last column and row undrawn. A TV's overscan
        // hides that, but with the Expand aspect ratio it shows as a dark strip at the window edge, so the previous,
        // full size is kept. The 3D scissor's right edge is also always the framebuffer width minus one (the game sets
        // 0x8010C506 to width - 1), including when in-mission cutscenes narrow its top and bottom; such a scissor
        // extends to the edge too.
        static void adjustScissor(State *state, uint32_t &w0, uint32_t &w1) {
            const int32_t ulx = (w0 >> 12) & 0xFFF;
            const int32_t uly = w0 & 0xFFF;
            int32_t lrx = (w1 >> 12) & 0xFFF;
            int32_t lry = w1 & 0xFFF;
            std::array<int32_t, 4> &previous = context.previousScissor;
            if ((ulx == previous[0]) && (uly == previous[1]) && (lrx == previous[2] - 4) && (lry == previous[3] - 4)) {
                lrx = previous[2];
                lry = previous[3];
            }
            else {
                previous = { ulx, uly, lrx, lry };
            }

            const int32_t fbRight = int32_t(state->rdp->colorImage.width) * 4;
            if ((fbRight > 4) && (lrx == fbRight - 4)) {
                lrx = fbRight;
            }

            w1 = (w1 & 0xFF000000U) | (uint32_t(lrx & 0xFFF) << 12) | uint32_t(lry & 0xFFF);
        }

        // Vertices are given as their byte offset in the vertex buffer divided by 8 (vertex n is n * 5).
        static bool vertexIndex(uint32_t encoded, uint32_t &index) {
            const uint32_t offset = encoded * 8;
            index = offset / VertexStride;
            return ((offset % VertexStride) == 0) && (index < VertexCount);
        }

        // Display list blocks.

        static void fetchBlock(State *state, DisplayList **dl, uint32_t address) {
            // Addresses are raw RDRAM addresses: there's no segment translation, only the DMA's address mask.
            address &= RSP_DMA_MASK;

            uint8_t *block = state->fromRDRAM(address);
            context.blockAddress = address;
            context.nextBlockAddress = *reinterpret_cast<const uint32_t *>(block);
            context.blockEnd = reinterpret_cast<DisplayList *>(block + BlockCommandsEnd);
            context.blockFetched = true;

            // The interpreter advances to the next command after the handler.
            *dl = reinterpret_cast<DisplayList *>(block + BlockHeaderSize) - 1;
        }

        // Runs after every command, like the microcode's check of the read pointer when a command finishes. It
        // doesn't apply when the command fetched a block itself (DL, BRANCH_DL, ENDDL, NEXT_BLOCK): the microcode
        // returns to the loop past the check in that case.
        static void advanceBlock(State *state, DisplayList **dl) {
            if (context.blockFetched) {
                context.blockFetched = false;
            }
            else if ((*dl != nullptr) && ((*dl + 1) == context.blockEnd)) {
                fetchBlock(state, dl, context.nextBlockAddress);
                context.blockFetched = false;
            }
        }

        static DisplayList *displayListStart(State *state, uint32_t dlStartAddress, DisplayList *dlStart) {
            // The task's display list points to the header of the first block.
            DisplayList *dl = dlStart;
            context.callStack.clear();
            fetchBlock(state, &dl, dlStartAddress);
            context.blockFetched = false;
            return dl + 1;
        }

        // Synchronizes the RT64 viewport and clip ratio with the ones in DMEM.

        static void syncViewport(State *state) {
            RSP *rsp = state->rsp.get();
            const std::array<int16_t, 7> values = {
                dmemReadS16(DmemViewportScale + 0), dmemReadS16(DmemViewportScale + 2), dmemReadS16(DmemViewportScale + 4),
                dmemReadS16(DmemViewportTranslate + 0), dmemReadS16(DmemViewportTranslate + 2), dmemReadS16(DmemViewportTranslate + 4),
                dmemReadS16(DmemClipRatio)
            };

            if (context.viewportSynced && (values == context.syncedViewport) && (context.viewportAnchor == context.syncedViewportAnchor)) {
                return;
            }

            // Widescreen: an anchored viewport keeps its contents at their distance from that edge. The translation is
            // given relative to the anchor so the viewport itself doesn't move in 4:3.
            uint16_t ori = rsp->extended.global.viewportOrigin;
            int32_t offsetX = rsp->extended.global.viewportOffsetX;
            if (context.viewportAnchor != G_EX_ORIGIN_NONE) {
                ori = context.viewportAnchor;
                offsetX = (ori == G_EX_ORIGIN_RIGHT) ? -int32_t(state->rdp->colorImage.width * 4) : 0;
            }

            // Same conversion as RSP::setViewport. The scale keeps its sign: the projection given to RT64 has y negated instead.
            interop::RSPViewport &viewport = rsp->viewportStack[rsp->viewportStackSize - 1];
            viewport.scale.x = float(values[0]) / 4.0f;
            viewport.scale.y = float(values[1]) / 4.0f;
            viewport.scale.z = float(values[2]) / DepthRange;
            viewport.translate.x = float(state->rdp->movedFromOrigin(values[3], ori) + offsetX) / 4.0f;
            viewport.translate.y = float(values[4] + rsp->extended.global.viewportOffsetY) / 4.0f;
            viewport.translate.z = float(values[5]) / DepthRange;
            rsp->extended.viewportOriginStack[rsp->viewportStackSize - 1] = ori;
            rsp->viewportChanged = true;

            // The guard band is the same on all four edges.
            if (rsp->clipRatios[0] != values[6]) {
                rsp->setClipRatioAll(values[6]);
            }

            context.syncedViewport = values;
            context.syncedViewportAnchor = context.viewportAnchor;
            context.viewportSynced = true;
        }

        // Vertices.

        // The end of the VTX command: transforms the vertex from the input at DMEM 0x280.
        static void transformVertex(uint32_t index, bool fog) {
            Vertex &v = context.vertices[index];
            const uint32_t inputAddress = DmemVertexInput + index * 8;
            const int64_t input[4] = { dmemReadS16(inputAddress + 0), dmemReadS16(inputAddress + 2), dmemReadS16(inputAddress + 4), 1 };
            for (uint32_t i = 0; i < 3; i++) {
                v.position[i] = int16_t(input[i]);
            }

            // Clip position: input x combined matrix.
            for (uint32_t j = 0; j < 4; j++) {
                int64_t sum = 0;
                for (uint32_t i = 0; i < 4; i++) {
                    sum += input[i] * dmemMatrixElement(DmemCombined, i, j);
                }

                v.clip[j] = clamp32(sum);
            }

            // Clip flags against w and against w * clip ratio.
            const int32_t w = v.clip[3];
            const int32_t guardW = clamp32(int64_t(w) * dmemReadS16(DmemClipRatio));
            uint16_t flags = 0;
            for (uint32_t i = 0; i < 3; i++) {
                bool below, above;
                clipTest(v.clip[i], w, below, above);
                flags |= below ? (0x10 << i) : 0;
                flags |= above ? (0x1000 << i) : 0;
                clipTest(v.clip[i], guardW, below, above);
                flags |= below ? (0x1 << i) : 0;
                flags |= above ? (0x100 << i) : 0;
            }

            v.clipFlags = flags;

            // 1/w' with w' = w * perspnorm. A negative 1/w' uses an integer part of 0x7FFF for the screen position.
            const uint16_t perspNorm = dmemRead16(DmemPerspNorm);
            v.invW = reciprocal(mulFixed(w, perspNorm));
            const int32_t screenInvW = (v.invW < 0) ? int32_t(0x7FFF0000U | (uint32_t(v.invW) & 0xFFFFU)) : v.invW;

            // Screen position: ndc * scale + translate, with ndc = clip * (1/w') * perspnorm. No y negation.
            int32_t ndc[3];
            for (uint32_t i = 0; i < 3; i++) {
                ndc[i] = mulFixed(mulFixed(v.clip[i], screenInvW), perspNorm);
                const int64_t screen = (int64_t(dmemReadS16(DmemViewportTranslate + i * 2)) << 16) + int64_t(ndc[i]) * dmemReadS16(DmemViewportScale + i * 2);
                v.screen[i] = clamp16(screen >> 16);
                if (i == 2) {
                    v.screenZFraction = uint16_t(screen);
                }
            }

            // Only the integer parts are clamped.
            v.screen[0] = std::max(v.screen[0], dmemReadS16(DmemScreenMinXY));
            v.screen[1] = std::max(v.screen[1], dmemReadS16(DmemScreenMinXY));
            v.screen[2] = std::max(v.screen[2], dmemReadS16(DmemScreenMinZ));

            // Fog goes to the alpha of the vertex color: clamp(floor((ndc z * multiplier + offset) * scale), 0, scale).
            if (fog) {
                const int32_t multiplier = dmemReadFixed(DmemFogMultiplier, DmemFogMultiplier + 2);
                const int32_t offset = dmemReadFixed(DmemFogOffset, DmemFogOffset + 2);
                const int16_t scale = dmemReadS16(DmemFogScale);
                const int32_t product = mulFixed(ndc[2], multiplier);
                const uint32_t fractionSum = (uint32_t(product) & 0xFFFF) + (uint32_t(offset) & 0xFFFF);
                const int16_t integerSum = clamp16(int64_t(product >> 16) + (offset >> 16) + int64_t(fractionSum >> 16));
                const int64_t value = (int64_t(integerSum) << 16) | (fractionSum & 0xFFFF);
                int16_t fogValue = clamp16((value * scale) >> 16);
                fogValue = std::max<int16_t>(fogValue, 0);
                fogValue = std::min<int16_t>(fogValue, scale);
                v.rgba = (v.rgba & 0xFFFFFF00U) | uint8_t(fogValue);
            }
        }

        // Transforms the vertices in the input area to the start of the vertex buffer. The microcode works on two
        // vertices at a time, so a count of 0 still transforms the first one. Returns the number of vertices.
        static uint32_t transformVertices(uint32_t count) {
            count = std::max(count, 1U);
            if (count > VertexCount) {
                RT64_LOG_PRINTF("F5Rogue: vertex count %u is larger than the vertex buffer", count);
                count = VertexCount;
            }

            const bool fog = (dmemRead32(DmemGeometryMode) & GeometryFog) != 0;
            for (uint32_t i = 0; i < count; i++) {
                transformVertex(i, fog);
            }

            return count;
        }

        // Gives RT64 the model space positions and the current attributes of the first vertices.
        static void loadVertices(State *state, uint32_t rdramAddress, uint32_t count) {
            std::array<RSP::Vertex, VertexCount> rspVertices = {};
            for (uint32_t i = 0; i < count; i++) {
                const Vertex &v = context.vertices[i];
                RSP::Vertex &dst = rspVertices[i];
                dst.x = v.position[0];
                dst.y = v.position[1];
                dst.z = v.position[2];
                dst.s = v.s;
                dst.t = v.t;
                dst.color.r = uint8_t(v.rgba >> 24);
                dst.color.g = uint8_t(v.rgba >> 16);
                dst.color.b = uint8_t(v.rgba >> 8);
                dst.color.a = uint8_t(v.rgba);
                context.loaded[i] = { v.rgba, v.s / 32.0f, v.t / 32.0f };
                context.screenShiftX[i] = 0.0f;
            }

            state->rsp->setVertexF5Rogue(rdramAddress, rspVertices.data(), count, 0);
            context.lastLoadAddress = rdramAddress;
            context.lastLoadCount = count;
        }

        // Widescreen.

        // Elements entirely within the outer 30% of the screen are anchored to that edge; returns G_EX_ORIGIN_NONE
        // otherwise. Coordinates in 10.2 like the RDP's.
        static uint16_t hudEdgeOrigin(int32_t left, int32_t right, int32_t fbWidth) {
            if (fbWidth <= 0) {
                return G_EX_ORIGIN_NONE;
            }

            const int32_t outerBand = (fbWidth * 3) / 10;
            if (right <= outerBand) {
                return G_EX_ORIGIN_LEFT;
            }
            else if (left >= (fbWidth - outerBand)) {
                return G_EX_ORIGIN_RIGHT;
            }
            else {
                return G_EX_ORIGIN_NONE;
            }
        }

        // Anchors the viewport for the following triangles. RT64 binds the viewport when vertices load, so the last
        // vertex batch is loaded again.
        static void setViewportAnchor(State *state, uint16_t origin) {
            if (origin == context.viewportAnchor) {
                return;
            }

            context.viewportAnchor = origin;
            syncViewport(state);
            if (context.lastLoadCount > 0) {
                loadVertices(state, context.lastLoadAddress, context.lastLoadCount);
            }
        }

        // Screen x of a vertex in pixels as the GPU computes it (before any widescreen move).
        static bool vertexScreenX(uint32_t index, float &x) {
            const Vertex &v = context.vertices[index];
            if (v.clip[3] <= 0) {
                return false;
            }

            const float ndcX = float(v.clip[0]) / float(v.clip[3]);
            x = ndcX * float(dmemReadS16(DmemViewportScale + 0)) / 4.0f + float(dmemReadS16(DmemViewportTranslate + 0)) / 4.0f;
            return true;
        }

        static bool vertexScreenY(uint32_t index, float &y) {
            const Vertex &v = context.vertices[index];
            if (v.clip[3] <= 0) {
                return false;
            }

            const float ndcY = float(v.clip[1]) / float(v.clip[3]);
            y = ndcY * float(dmemReadS16(DmemViewportScale + 2)) / 4.0f + float(dmemReadS16(DmemViewportTranslate + 2)) / 4.0f;
            return true;
        }

        // Rogue Squadron's radar map triangles.
        constexpr uint32_t RadarOtherModeL = 0x00504240U;
        constexpr uint32_t RadarGeometryMode = 0x00801005U;

        // Front-end menu wipe mesh.
        constexpr uint32_t WipeOtherModeL = 0x00504240U;
        constexpr uint32_t WipeFramebufferWidth = 512;

        // Picks the viewport anchor for a TRI1/TRI2 command's vertices and closes the menu wipe's gaps.
        static void widescreenTriangles(State *state, const uint32_t *vertices, uint32_t vertexCount, bool hasTexcoords) {
            const interop::OtherMode &otherMode = state->rdp->otherMode;
            const uint32_t geometryMode = dmemRead32(DmemGeometryMode);
            uint16_t origin = G_EX_ORIGIN_NONE;

            // Full-screen overlays (pause darkening, fades): untextured geometry without depth test that covers the
            // full height and reaches a left or right edge stretches across the wide screen.
            if (!otherMode.zCmp() && !hasTexcoords) {
                float minX = FLT_MAX, maxX = -FLT_MAX, minY = FLT_MAX, maxY = -FLT_MAX;
                bool valid = true;
                for (uint32_t i = 0; (i < vertexCount) && valid; i++) {
                    float x, y;
                    valid = vertexScreenX(vertices[i], x) && vertexScreenY(vertices[i], y);
                    minX = std::min(minX, x);
                    maxX = std::max(maxX, x);
                    minY = std::min(minY, y);
                    maxY = std::max(maxY, y);
                }

                // Screen extent from the current viewport (pixels).
                const float scaleX = std::abs(float(dmemReadS16(DmemViewportScale + 0))) / 4.0f;
                const float scaleY = std::abs(float(dmemReadS16(DmemViewportScale + 2))) / 4.0f;
                const float translateX = float(dmemReadS16(DmemViewportTranslate + 0)) / 4.0f;
                const float translateY = float(dmemReadS16(DmemViewportTranslate + 2)) / 4.0f;
                const float left = translateX - scaleX, right = translateX + scaleX;
                const float top = translateY - scaleY, bottom = translateY + scaleY;
                if (valid && (right > left) && (minY <= top + 1.0f) && (maxY >= bottom - 2.0f) && ((minX <= left + 1.0f) || (maxX >= right - 1.0f))) {
                    origin = ViewportOriginStretch;
                }
            }

            // The radar map: 8 untested-depth triangles with exactly these modes. In an outer band of the screen they
            // anchor to that edge so the map moves with the radar ring (texture rectangles, see drawTexRect). Other
            // geometry without depth test (sky, effects) isn't matched.
            if ((origin == G_EX_ORIGIN_NONE) && !otherMode.zCmp() && (otherMode.L == RadarOtherModeL) && (geometryMode == RadarGeometryMode)) {
                float minX = FLT_MAX, maxX = -FLT_MAX;
                bool valid = true;
                for (uint32_t i = 0; (i < vertexCount) && valid; i++) {
                    float x;
                    valid = vertexScreenX(vertices[i], x);
                    minX = std::min(minX, x);
                    maxX = std::max(maxX, x);
                }

                if (valid) {
                    origin = hudEdgeOrigin(int32_t(minX * 4.0f), int32_t(maxX * 4.0f), int32_t(state->rdp->colorImage.width) * 4);
                }
            }

            setViewportAnchor(state, origin);

            // Front-end wipes (title <-> menus) draw both pictures as 64-pixel texture rectangles on the exact 64-pixel
            // grid and only the soft band as this mesh, whose tiles are 63.91 pixels wide. In a right-to-left wipe the
            // mesh's opaque right end stops up to 0.66 pixels short of the next rectangle (e.g. 447.43 against 448),
            // and the picture underneath shows through that sliver. The N64 fills a pixel when a span covers its left
            // edge, so it never shows; at higher resolutions it is a column of several pixels. Opaque right edges move
            // onto the grid. The mesh's tile columns also meet the screen edges a fraction of a pixel inside, so
            // corners within one pixel of a screen edge snap onto it. Only this exact mode is affected.
            if (hasTexcoords && (otherMode.textPersp() != G_TP_PERSP) && !otherMode.zCmp() && (otherMode.L == WipeOtherModeL) &&
                (state->rdp->colorImage.width == WipeFramebufferWidth))
            {
                float cornerX[4] = {};
                float maxX = -FLT_MAX;
                for (uint32_t i = 0; i < vertexCount; i++) {
                    if (!vertexScreenX(vertices[i], cornerX[i])) {
                        return;
                    }

                    cornerX[i] += context.screenShiftX[vertices[i]];
                    maxX = std::max(maxX, cornerX[i]);
                }

                auto shift = [state](uint32_t index, float dx) {
                    state->rsp->shiftVertexScreenX(uint16_t(index), dx);
                    context.screenShiftX[index] += dx;
                };

                const float GridSize = 64.0f;
                const float target = std::ceil(maxX / GridSize) * GridSize;
                const float dx = target - maxX;
                if ((dx > 0.001f) && (dx < 1.0f)) {
                    for (uint32_t i = 0; i < vertexCount; i++) {
                        const bool opaque = (context.vertices[vertices[i]].rgba & 0xFF) == 0xFF;
                        if (opaque && (std::abs(cornerX[i] - maxX) < 0.01f)) {
                            shift(vertices[i], dx);
                            cornerX[i] += dx;
                        }
                    }
                }

                const float screenRight = float(state->rdp->colorImage.width);
                for (uint32_t i = 0; i < vertexCount; i++) {
                    if ((cornerX[i] > 0.0f) && (cornerX[i] < 1.0f)) {
                        shift(vertices[i], -cornerX[i]);
                    }
                    else if ((cornerX[i] > screenRight - 1.0f) && (cornerX[i] < screenRight)) {
                        shift(vertices[i], screenRight - cornerX[i]);
                    }
                }
            }
        }

        // Triangles.

        static void syncDrawState(State *state) {
            RSP *rsp = state->rsp.get();
            const uint32_t geometryMode = dmemRead32(DmemGeometryMode);
            const uint32_t rt64GeometryMode = geometryMode & GeometryPassedToRT64;
            if (rsp->geometryModeStack[rsp->geometryModeStackSize - 1] != rt64GeometryMode) {
                rsp->modifyGeometryMode(0, rt64GeometryMode);
            }

            // Texturing is enabled by geometry bit 1 (which TEXTURE sets from its w0 bit 0). The tile and level come
            // from the last TEXTURE command; its w1 (the F3D texture scale) is ignored.
            const uint32_t texture = dmemRead32(DmemTexture);
            rsp->setTexture((texture >> 8) & 0x7, (texture >> 11) & 0x7, (geometryMode & GeometryTexture) ? 1 : 0, 0, 0);
        }

        // Changes RT64's copy of the vertex to the attributes it has for this triangle.
        static void prepareVertex(State *state, uint32_t index, uint32_t colorIndex, bool shade, bool texture, float texcoordScale) {
            const Vertex &v = context.vertices[index];
            LoadedAttributes &loaded = context.loaded[index];
            const uint32_t rgba = shade ? context.vertices[colorIndex].rgba : loaded.rgba;
            const float s = texture ? (v.s * texcoordScale) : loaded.s;
            const float t = texture ? (v.t * texcoordScale) : loaded.t;
            if ((rgba != loaded.rgba) || (s != loaded.s) || (t != loaded.t)) {
                state->rsp->setVertexColorTexcoord(uint16_t(index), rgba, s, t);
                loaded = { rgba, s, t };
            }
        }

        static void debugDumpTriangle(State *state, uint32_t a, uint32_t b, uint32_t c, uint32_t flatIndex, uint32_t geometryMode, bool clipped) {
            FILE *f = debugDump();
            if (f == nullptr) {
                return;
            }

            uint32_t order[3] = { a, b, c };
            std::stable_sort(std::begin(order), std::end(order), [](uint32_t p, uint32_t q) {
                return context.vertices[p].screen[1] < context.vertices[q].screen[1];
            });

            fprintf(f, "tri\t%u\t%02X\t%u", debugDumpIndex++, 0xC8 | (geometryMode & 0x7), (dmemRead32(DmemTexture) >> 8) & 0x7);
            for (uint32_t index : order) {
                const Vertex &v = context.vertices[index];
                const uint32_t rgba = context.vertices[(flatIndex == UINT32_MAX) ? index : flatIndex].rgba;
                const double z = (double(v.screen[2]) + v.screenZFraction / 65536.0) * 32.0;
                fprintf(f, "\t%.3f %.3f %.3f %u %u %u %u %d %d 0", v.screen[0] / 4.0, v.screen[1] / 4.0, z,
                    rgba >> 24, (rgba >> 16) & 0xFF, (rgba >> 8) & 0xFF, rgba & 0xFF, v.s, v.t);
            }

            fprintf(f, "\t%s\n", clipped ? "clipped" : "");
            (void)state;
        }

        // The triangle setup of the microcode. flatIndex is the vertex whose color is used without smooth shading.
        static void drawTriangle(State *state, uint32_t a, uint32_t b, uint32_t c, uint32_t flatIndex) {
            const Vertex &va = context.vertices[a];
            const Vertex &vb = context.vertices[b];
            const Vertex &vc = context.vertices[c];
            const uint32_t geometryMode = dmemRead32(DmemGeometryMode);

            // All three vertices are outside of the same frustum plane.
            if ((va.clipFlags & vb.clipFlags & vc.clipFlags & ClipRejectMask) != 0) {
                return;
            }

            // Winding, with the microcode's sign: negative is clockwise on screen (y down).
            double cross;
            if (((va.clipFlags | vb.clipFlags | vc.clipFlags) & ClipRequiredMask) != 0) {
                // The microcode clips this triangle and tests each triangle of the result. Their winding is the one of
                // the visible part, which is given by the determinant of the clip space (x, y, w) of the vertices.
                auto clipValue = [](const Vertex &v, uint32_t i) {
                    return double(v.clip[i]) / 65536.0;
                };

                const double determinant =
                    clipValue(va, 0) * (clipValue(vb, 1) * clipValue(vc, 3) - clipValue(vb, 3) * clipValue(vc, 1)) -
                    clipValue(va, 1) * (clipValue(vb, 0) * clipValue(vc, 3) - clipValue(vb, 3) * clipValue(vc, 0)) +
                    clipValue(va, 3) * (clipValue(vb, 0) * clipValue(vc, 1) - clipValue(vb, 1) * clipValue(vc, 0));

                const double scaleSign = double(dmemReadS16(DmemViewportScale + 0)) * double(dmemReadS16(DmemViewportScale + 2));
                cross = -scaleSign * determinant;
            }
            else {
                // Screen positions in quarter pixels. The differences saturate to 16 bits like VSUB.
                auto difference = [](int16_t p, int16_t q) {
                    return int64_t(clamp16(int32_t(p) - q));
                };

                cross = double(difference(vc.screen[0], va.screen[0]) * difference(vb.screen[1], va.screen[1]) -
                    difference(vb.screen[0], va.screen[0]) * difference(vc.screen[1], va.screen[1]));
            }

            // Degenerate triangles are dropped. Bit 13 culls clockwise triangles, there is no front face culling.
            if (cross == 0.0) {
                return;
            }

            if ((cross < 0.0) && ((geometryMode & GeometryCullClockwise) != 0)) {
                return;
            }

            // Without bit 12, s and t are divided by w like F3D does. With it they go to the RDP as they are. With
            // texture perspective off in the other mode, the RDP then uses them directly, while RT64 halves the texture
            // coordinates in that case to match F3D (whose s and t are divided by w), so they're doubled here.
            float texcoordScale = 1.0f / 32.0f;
            if ((geometryMode & GeometryNoPerspective) && (state->rdp->otherMode.textPersp() != G_TP_PERSP)) {
                texcoordScale *= 2.0f;
            }

            // Smooth shading uses the color of each vertex, flat shading the RGBA of one vertex for all three.
            const bool shade = (geometryMode & GeometryShade) != 0;
            const bool texture = (geometryMode & GeometryTexture) != 0;
            const bool smooth = (geometryMode & GeometrySmooth) != 0;
            prepareVertex(state, a, smooth ? a : flatIndex, shade, texture, texcoordScale);
            prepareVertex(state, b, smooth ? b : flatIndex, shade, texture, texcoordScale);
            prepareVertex(state, c, smooth ? c : flatIndex, shade, texture, texcoordScale);
            syncDrawState(state);
            debugDumpTriangle(state, a, b, c, smooth ? UINT32_MAX : flatIndex, geometryMode,
                ((va.clipFlags | vb.clipFlags | vc.clipFlags) & ClipRequiredMask) != 0);
            state->rsp->drawIndexedTri(a, b, c);
        }

        static void triangleCommand(State *state, DisplayList **dl, bool fourVertices) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t w1 = (*dl)->w1;
            const uint32_t colorIndices = (*dl)[1].w0;
            const bool hasTexcoords = (w0 & 0x2) != 0;
            const bool fogFourthVertex = (w0 & 0x4) != 0;
            *dl = *dl + (hasTexcoords ? 3 : 1);

            uint32_t a, b, c, d = 0;
            if (!vertexIndex((w1 >> 16) & 0xFF, a) || !vertexIndex((w1 >> 8) & 0xFF, b) || !vertexIndex(w1 & 0xFF, c) || (fourVertices && !vertexIndex(w1 >> 24, d))) {
                RT64_LOG_PRINTF("F5Rogue: invalid triangle vertices 0x%08X", w1);
                return;
            }

            // Byte offsets into the color table.
            const uint32_t colorA = DmemColorTable + ((colorIndices >> 16) & 0xFF);
            const uint32_t colorB = DmemColorTable + ((colorIndices >> 8) & 0xFF);
            const uint32_t colorC = DmemColorTable + (colorIndices & 0xFF);
            const uint32_t colorD = DmemColorTable + (colorIndices >> 24);

            // With fog, the fog value of each vertex is first written to the alpha of the color table entry it uses
            // (the entry stays changed). For TRI1, the fourth "vertex" is at DMEM 0.
            Vertex &va = context.vertices[a];
            Vertex &vb = context.vertices[b];
            Vertex &vc = context.vertices[c];
            Vertex &vd = context.vertices[d];
            if (dmemRead32(DmemGeometryMode) & GeometryFog) {
                const uint8_t fogA = uint8_t(va.rgba);
                const uint8_t fogB = uint8_t(vb.rgba);
                const uint8_t fogC = uint8_t(vc.rgba);
                const uint8_t fogD = fourVertices ? uint8_t(vd.rgba) : dmemRead8(0x13);
                dmemWrite8(colorA + 3, fogA);
                dmemWrite8(colorB + 3, fogB);
                dmemWrite8(colorC + 3, fogC);
                if (fogFourthVertex) {
                    dmemWrite8(colorD + 3, fogD);
                }
            }

            // Stored in the same order as the microcode in case a vertex is used twice.
            const uint32_t rgbaC = dmemRead32(colorC);
            const uint32_t rgbaD = dmemRead32(colorD);
            va.rgba = dmemRead32(colorA);
            vb.rgba = dmemRead32(colorB);

            int16_t texcoords[8] = {};
            if (hasTexcoords) {
                // Each s, t is multiplied by the texture scale from MOVEMEM 0x82 (entries s, t, s, t): a and c use the
                // first two entries, b and d the last two.
                const DisplayList *texcoordWords = *dl - 1;
                const uint32_t words[4] = { texcoordWords[0].w0, texcoordWords[0].w1, texcoordWords[1].w0, texcoordWords[1].w1 };
                for (uint32_t i = 0; i < 8; i++) {
                    const int16_t value = int16_t(words[i / 2] >> ((i & 1) ? 0 : 16));
                    const uint32_t entry = (i & 3) * 2;
                    const int32_t scale = dmemReadFixed(DmemTexcoordScaleInteger + entry, DmemTexcoordScaleFraction + entry);
                    texcoords[i] = clamp16((int64_t(value) * scale) >> 16);
                }

                va.s = texcoords[0];
                va.t = texcoords[1];
                vb.s = texcoords[2];
                vb.t = texcoords[3];
                vc.s = texcoords[4];
                vc.t = texcoords[5];
            }

            vc.rgba = rgbaC;

            if (fourVertices) {
                vd.rgba = rgbaD;

                // TRI2 always stores the fourth s, t, even without texture coordinates in the command (it is then
                // whatever was left in the vector register). That case keeps the previous s, t here.
                if (hasTexcoords) {
                    vd.s = texcoords[6];
                    vd.t = texcoords[7];
                }
            }

            const uint32_t commandVertices[4] = { a, b, c, d };
            widescreenTriangles(state, commandVertices, fourVertices ? 4 : 3, hasTexcoords);

            drawTriangle(state, a, b, c, a);

            if (fourVertices) {
                drawTriangle(state, a, c, d, a);
            }
        }

        // TRI_GEN.

        // A grid point as five 16-bit lanes: the height and the four color bytes.
        struct GridPoint {
            int32_t lanes[5];
        };

        static GridPoint loadGridPoint(uint32_t k) {
            GridPoint p;
            p.lanes[0] = dmemReadS16(DmemGridHeights + k * 2);
            for (uint32_t i = 0; i < 4; i++) {
                p.lanes[1 + i] = dmemRead8(DmemColorTable + k * 4 + i);
            }

            return p;
        }

        static void storeGridPoint(uint32_t k, const GridPoint &p) {
            dmemWrite16(DmemGridHeights + k * 2, uint16_t(p.lanes[0]));
            for (uint32_t i = 0; i < 4; i++) {
                dmemWrite8(DmemColorTable + k * 4 + i, uint8_t(p.lanes[1 + i]));
            }
        }

        // floor((a + b) / 2), with the sum wrapped to 16 bits.
        static GridPoint gridAverage(const GridPoint &a, const GridPoint &b) {
            GridPoint r;
            for (uint32_t i = 0; i < 5; i++) {
                r.lanes[i] = clamp16((int64_t(int16_t(a.lanes[i] + b.lanes[i])) * 0x8000) >> 16);
            }

            return r;
        }

        // floor(x * w + p * (1 - w)) with x = a * fa + b * fb kept with 16 fraction bits. Weights are 0.16 fractions.
        static GridPoint gridBlend(const GridPoint &a, uint32_t fa, const GridPoint &b, uint32_t fb, const GridPoint &p, uint32_t w) {
            const int64_t inverseW = (0x10000 - w) & 0xFFFF;
            GridPoint r;
            for (uint32_t i = 0; i < 5; i++) {
                const int64_t x = int64_t(a.lanes[i]) * fa + int64_t(b.lanes[i]) * fb;
                const int64_t xInteger = clamp16(x >> 16);
                const int64_t xFraction = x & 0xFFFF;
                const int64_t sum = ((xFraction * w) >> 16) + xInteger * w + int64_t(p.lanes[i]) * inverseW;
                r.lanes[i] = clamp16(sum >> 16);
            }

            return r;
        }

        // floor((a + b) * (w / 2) + p * (1 - w)), with the sum wrapped to 16 bits and w / 2 truncated.
        static GridPoint gridInterior(const GridPoint &a, const GridPoint &b, const GridPoint &p, uint32_t w) {
            const int64_t halfW = w >> 1;
            const int64_t inverseW = (0x10000 - w) & 0xFFFF;
            GridPoint r;
            for (uint32_t i = 0; i < 5; i++) {
                const int64_t sum = int64_t(int16_t(a.lanes[i] + b.lanes[i])) * halfW + int64_t(p.lanes[i]) * inverseW;
                r.lanes[i] = clamp16(sum >> 16);
            }

            return r;
        }

        struct TriGenCommand {
            uint32_t words[TriGenSize / 4];

            uint8_t byte(uint32_t offset) const {
                return uint8_t(words[offset / 4] >> (24 - (offset % 4) * 8));
            }

            uint16_t half(uint32_t offset) const {
                return uint16_t(words[offset / 4] >> (16 - (offset % 4) * 8));
            }

            uint32_t word(uint32_t offset) const {
                return words[offset / 4];
            }
        };

        // Depth sort list. The nodes are 6 bytes at DMEM 0xB70 + offset (the color table): +0 next, +1 previous,
        // +2 key, +4 triangle id. The head is the byte at 0xC70. This is the microcode's insertion procedure, with
        // the cursor left at the inserted node.
        struct SortList {
            uint32_t cursor = 0;
            uint32_t newNode = 2;
            bool inverted = false;

            static uint8_t next(uint32_t node) { return dmemRead8(DmemColorTable + node + 0); }
            static uint8_t previous(uint32_t node) { return dmemRead8(DmemColorTable + node + 1); }
            static int16_t key(uint32_t node) { return dmemReadS16(DmemColorTable + node + 2); }
            static void setNext(uint32_t node, uint32_t value) { dmemWrite8(DmemColorTable + node + 0, uint8_t(value)); }
            static void setPrevious(uint32_t node, uint32_t value) { dmemWrite8(DmemColorTable + node + 1, uint8_t(value)); }
            static void setHead(uint32_t node) { dmemWrite8(DmemSortHead, uint8_t(node)); }

            void insert(int16_t newKey, uint8_t id) {
                const uint32_t node = newNode & 0xFF;
                if (cursor == 0) {
                    setNext(node, 0);
                    setPrevious(node, 0);
                    setHead(node);
                }
                else if (key(cursor) < newKey) {
                    // Walk toward the head.
                    while (true) {
                        const uint32_t previousNode = previous(cursor);
                        if (previousNode == 0) {
                            // Insert before the cursor as the new head.
                            setNext(node, cursor);
                            setPrevious(node, 0);
                            setPrevious(cursor, node);
                            setHead(node);
                            break;
                        }

                        cursor = previousNode;
                        if (!((key(cursor) < newKey) != inverted)) {
                            // Insert between the cursor and its next node.
                            const uint32_t nextNode = next(cursor);
                            setPrevious(node, cursor);
                            setNext(cursor, node);
                            setNext(node, nextNode);
                            if (nextNode != 0) {
                                setPrevious(nextNode, node);
                            }

                            break;
                        }
                    }
                }
                else {
                    // Walk toward the tail.
                    while (true) {
                        const uint32_t nextNode = next(cursor);
                        if (nextNode == 0) {
                            // Append after the cursor.
                            setPrevious(node, cursor);
                            setNext(cursor, node);
                            setNext(node, 0);
                            break;
                        }

                        cursor = nextNode;
                        if ((key(cursor) < newKey) != inverted) {
                            // Insert between the cursor and its previous node.
                            const uint32_t previousNode = previous(cursor);
                            setNext(node, cursor);
                            setPrevious(node, previousNode);
                            setPrevious(cursor, node);
                            if (previousNode != 0) {
                                setNext(previousNode, node);
                            }
                            else {
                                setHead(node);
                            }

                            break;
                        }
                    }
                }

                cursor = node;
                dmemWrite8(DmemColorTable + node + 4, id);
                dmemWrite16(DmemColorTable + node + 2, uint16_t(newKey));
                newNode += 6;
            }
        };

        // floor((z_a + z_b + z_c) * 0x5555 / 65536) with the screen z integer parts.
        static int16_t triGenSortKey(uint32_t a, uint32_t b, uint32_t c) {
            const int64_t sum = int64_t(context.vertices[a].screen[2]) + context.vertices[b].screen[2] + context.vertices[c].screen[2];
            return clamp16((sum * dmemRead16(DmemSortKeyScale)) >> 16);
        }

        static void writeVertexInput(uint32_t index, int16_t x, int16_t y, int16_t z) {
            const uint32_t address = DmemVertexInput + index * 8;
            dmemWrite16(address + 0, uint16_t(x));
            dmemWrite16(address + 2, uint16_t(y));
            dmemWrite16(address + 4, uint16_t(z));
        }

        static void triGenQuad(State *state, const TriGenCommand &cmd) {
            const int16_t x = int16_t(cmd.half(0x20));
            const int16_t y = int16_t(cmd.half(0x22) << 4);
            const int16_t z = int16_t(cmd.half(0x24));
            const int16_t size = int16_t(cmd.half(0x26));
            const int16_t step = int16_t(cmd.half(0x1E));
            const bool inverted = (cmd.byte(2) & 0x1) != 0;

            // Corners (0, 0), (size, 0), (0, size), (size, size) with their height offsets.
            writeVertexInput(0, x, int16_t(y + cmd.half(0x04)), z);
            writeVertexInput(1, int16_t(x + size), int16_t(y + cmd.half(0x06)), z);
            writeVertexInput(2, x, int16_t(y + cmd.half(0x08)), int16_t(z + size));
            writeVertexInput(3, int16_t(x + size), int16_t(y + cmd.half(0x0A)), int16_t(z + size));

            const uint32_t count = transformVertices(4);

            // Colors c0-c3 (alpha keeps the vertex's fog value) and texture coordinates without the texture scale.
            const int16_t texcoords[4][2] = { { 0, step }, { step, step }, { 0, 0 }, { step, 0 } };
            for (uint32_t i = 0; i < 4; i++) {
                Vertex &v = context.vertices[i];
                v.rgba = (cmd.word(0x0C + i * 4) & 0xFFFFFF00U) | (v.rgba & 0xFF);
                v.s = texcoords[i][0];
                v.t = texcoords[i][1];
            }

            loadVertices(state, 0, count);

            // The order of the two halves depends on the screen z of vertices 1 and 2. Vertex 0 is the flat shading vertex.
            if ((context.vertices[1].screen[2] < context.vertices[2].screen[2]) != inverted) {
                drawTriangle(state, 0, 1, 3, 0);
                drawTriangle(state, 0, 3, 2, 0);
            }
            else {
                drawTriangle(state, 0, 3, 2, 0);
                drawTriangle(state, 0, 1, 3, 0);
            }
        }

        // Grid edges in the order the microcode processes them, with the crack fillers for when the neighbor's LOD is
        // different (vertex indices for the 5x5 layout, used when s is 0, and the 3x3 one).
        struct TriGenEdge {
            uint32_t lodOffset;
            uint32_t weightOffset;
            bool column;
            bool last;
            uint8_t fillers5x5[2][3];
            uint8_t fillers3x3[3];
        };

        static const TriGenEdge TriGenEdges[4] = {
            { 0x06, 0x1A, false, false, { { 0, 10, 5 }, { 10, 20, 15 } }, { 0, 6, 3 } },
            { 0x07, 0x1C, false, true, { { 4, 9, 14 }, { 14, 19, 24 } }, { 2, 5, 8 } },
            { 0x04, 0x16, true, false, { { 0, 1, 2 }, { 2, 3, 4 } }, { 0, 1, 2 } },
            { 0x05, 0x18, true, true, { { 20, 22, 21 }, { 22, 24, 23 } }, { 6, 8, 7 } }
        };

        static void triGenGrid(State *state, const TriGenCommand &cmd) {
            const uint32_t n = cmd.byte(1);
            const uint32_t shift = cmd.byte(3);
            const uint32_t colorShift = (cmd.byte(2) >> 4) & 0xF;
            const bool inverted = (cmd.byte(2) & 0x1) != 0;
            const uint32_t side = ((n - 1) >> (shift & 31)) + 1;
            if ((side < 2) || (side > TriGenMaxSide)) {
                RT64_LOG_PRINTF("F5Rogue: unsupported TRI_GEN grid (n %u, s %u)", n, shift);
                return;
            }

            // The height and color maps are copied to DMEM like the microcode does, so reads past the given sizes see
            // the same data.
            const uint32_t heightAddress = cmd.word(0x08);
            const uint32_t colorAddress = cmd.word(0x0C);
            dmaRead(state, DmemGridHeightMap, heightAddress, ((cmd.half(0x10) + 0x16U) & 0xFF0) - 1);
            dmaRead(state, DmemGridColorMap, colorAddress, ((cmd.half(0x12) + 0x13U) & 0xFF0) - 1);
            const uint32_t heightBase = DmemGridHeightMap + (heightAddress & 0x7);
            const uint32_t colorBase = DmemGridColorMap + (colorAddress & 0x7);

            // Point (r, c) is k = r * side + c: its height goes to DMEM 0xCB4 + k * 2 and its color to the color table
            // entry k (the color table stays changed). The color map can have a lower resolution than the grid.
            const uint32_t colorWidth = ((n - 1) >> (colorShift & 31)) + 1;
            const int32_t shiftDifference = int32_t(colorShift) - int32_t(shift);
            uint32_t colorStep, colorMask;
            if (shiftDifference >= 0) {
                colorStep = 0x100U >> (shiftDifference & 31);
                colorMask = ~(0xFFFFFFFFU << (shiftDifference & 31));
            }
            else {
                colorStep = 0x100U << ((-shiftDifference) & 31);
                colorMask = 0;
            }

            uint32_t rowPosition = 0;
            for (uint32_t r = 0; r < side; r++) {
                // Color map position in 8.8: ((r * step) >> 8) * width + ((c * step) >> 8).
                uint32_t position = uint32_t(int32_t(clamp16(int64_t(int16_t(rowPosition & 0xFF00)) * int16_t(colorWidth))));
                for (uint32_t c = 0; c < side; c++) {
                    const uint32_t entry = (position >> 8) * 4;
                    uint32_t neighbor = entry;
                    if (c & colorMask) {
                        neighbor += 4;
                    }

                    if (r & colorMask) {
                        neighbor += colorWidth * 4;
                    }

                    // Points between entries average the entry with the one to the right, below or diagonal.
                    const uint32_t k = r * side + c;
                    if ((c | r) & colorMask) {
                        for (uint32_t i = 0; i < 4; i++) {
                            const uint32_t sum = uint32_t(dmemRead8(colorBase + entry + i)) + dmemRead8(colorBase + neighbor + i);
                            dmemWrite8(DmemColorTable + k * 4 + i, uint8_t(sum >> 1));
                        }
                    }
                    else {
                        dmemWrite32(DmemColorTable + k * 4, dmemRead32(colorBase + entry));
                    }

                    const int8_t height = int8_t(dmemRead8(heightBase + r * (n << (shift & 31)) + c * (1U << (shift & 31))));
                    dmemWrite16(DmemGridHeights + k * 2, uint16_t(int32_t(height) * 16));
                    position += colorStep;
                }

                rowPosition += colorStep;
            }

            // Edges: match the neighbor's LOD and blend toward it ("ponderation").
            for (const TriGenEdge &edge : TriGenEdges) {
                const uint32_t base = edge.last ? (edge.column ? (side - 1) : ((side - 1) * side)) : 0;
                const uint32_t stride = edge.column ? side : 1;
                auto point = [&](uint32_t i) {
                    return base + i * stride;
                };

                const uint32_t weight = cmd.half(edge.weightOffset);
                if (cmd.byte(edge.lodOffset) != shift) {
                    for (uint32_t i = 1; i < side; i += 2) {
                        storeGridPoint(point(i), gridAverage(loadGridPoint(point(i - 1)), loadGridPoint(point(i + 1))));
                    }

                    // Points between every fourth point are blended toward the line between them. On a 3-point edge
                    // this reads past the edge, like the microcode.
                    if (weight != 0) {
                        for (uint32_t i = 1; (i + 1) < side; i++) {
                            if ((i & 3) != 0) {
                                const uint32_t factor = (i & 3) << 14;
                                const uint32_t i0 = i & ~3U;
                                const GridPoint lerp0 = loadGridPoint(point(i0));
                                const GridPoint lerp1 = loadGridPoint(point(i0 + 4));
                                storeGridPoint(point(i), gridBlend(lerp0, (0x10000 - factor) & 0xFFFF, lerp1, factor, loadGridPoint(point(i)), weight));
                            }
                        }
                    }
                }
                else if (weight != 0) {
                    for (uint32_t i = 1; i < side; i += 2) {
                        const GridPoint p0 = loadGridPoint(point(i - 1));
                        const GridPoint p1 = loadGridPoint(point(i + 1));
                        storeGridPoint(point(i), gridBlend(p0, 0x8000, p1, 0x8000, loadGridPoint(point(i)), weight));
                    }
                }
            }

            // Interior points with an odd row or column are blended toward their two neighbors along the row, the
            // column or the diagonal.
            const uint32_t interiorWeight = cmd.half(0x14);
            if ((interiorWeight != 0) && (side >= 3)) {
                for (uint32_t r = 1; (r + 1) < side; r++) {
                    for (uint32_t c = 1; (c + 1) < side; c++) {
                        if (((r | c) & 1) != 0) {
                            const uint32_t offset = ((r & 1) ? side : 0) + ((c & 1) ? 1 : 0);
                            const uint32_t k = r * side + c;
                            storeGridPoint(k, gridInterior(loadGridPoint(k - offset), loadGridPoint(k + offset), loadGridPoint(k), interiorWeight));
                        }
                    }
                }
            }

            // Vertices are column-major: point (r, c) is vertex c * side + r.
            const int16_t x = int16_t(cmd.half(0x20));
            const int16_t y = int16_t(cmd.half(0x22) << 4);
            const int16_t z = int16_t(cmd.half(0x24));
            const int16_t size = int16_t(cmd.half(0x26));
            for (uint32_t c = 0; c < side; c++) {
                for (uint32_t r = 0; r < side; r++) {
                    const int16_t height = dmemReadS16(DmemGridHeights + (r * side + c) * 2);
                    writeVertexInput(c * side + r, int16_t(x + int32_t(c) * size), int16_t(y + height), int16_t(z + int32_t(r) * size));
                }
            }

            const uint32_t count = transformVertices(side * side);

            // Colors (alpha keeps the vertex's fog value) and texture coordinates without the texture scale.
            const int16_t step = int16_t(cmd.half(0x1E));
            const int16_t lastT = clamp16(int64_t(step) * int32_t(side - 1));
            for (uint32_t c = 0; c < side; c++) {
                for (uint32_t r = 0; r < side; r++) {
                    Vertex &v = context.vertices[c * side + r];
                    v.rgba = (dmemRead32(DmemColorTable + (r * side + c) * 4) & 0xFFFFFF00U) | (v.rgba & 0xFF);
                    v.s = int16_t(int32_t(c) * step);
                    v.t = int16_t(lastT - int32_t(r) * step);
                }
            }

            loadVertices(state, 0, count);

            // Sort the triangles: cells (column outer, row inner, two triangles each) and then the crack fillers.
            // Cell triangles are identified by column * 4 + row (plus 0x40 for the second one), fillers by 0x80 plus
            // six times their index.
            SortList sortList;
            sortList.inverted = inverted;
            for (uint32_t c = 0; (c + 1) < side; c++) {
                for (uint32_t r = 0; (r + 1) < side; r++) {
                    const uint32_t v00 = c * side + r;
                    const uint32_t v10 = v00 + side;
                    const uint32_t v11 = v10 + 1;
                    const uint32_t v01 = v00 + 1;
                    const uint8_t id = uint8_t(c * 4 + r);
                    sortList.insert(triGenSortKey(v00, v10, v11), id);
                    sortList.insert(triGenSortKey(v00, v11, v01), id | TriGenSecondTriangle);
                }
            }

            std::array<std::array<uint8_t, 3>, 8> fillers = {};
            uint32_t fillerCount = 0;
            for (const TriGenEdge &edge : TriGenEdges) {
                if (cmd.byte(edge.lodOffset) == shift) {
                    continue;
                }

                const uint32_t edgeFillerCount = (shift == 0) ? 2 : 1;
                for (uint32_t f = 0; f < edgeFillerCount; f++) {
                    const uint8_t *triple = (shift == 0) ? edge.fillers5x5[f] : edge.fillers3x3;
                    fillers[fillerCount] = { triple[0], triple[1], triple[2] };
                    sortList.insert(triGenSortKey(triple[0], triple[1], triple[2]), uint8_t(TriGenFiller | (fillerCount * 6)));
                    fillerCount++;
                }
            }

            // Draw from the head along the next links. The first vertex is the flat shading vertex.
            uint32_t node = dmemRead8(DmemSortHead);
            for (uint32_t i = 0; (node != 0) && (i < 0x100); i++) {
                const uint8_t id = dmemRead8(DmemColorTable + node + 4);
                node = SortList::next(node);

                uint32_t a, b, c;
                if ((id >> 6) == 2) {
                    const std::array<uint8_t, 3> &filler = fillers[((id & 0x7F) / 6) & 0x7];
                    a = filler[0];
                    b = filler[1];
                    c = filler[2];
                }
                else {
                    a = ((id >> 2) & 0x3) * side + (id & 0x3);
                    if (id & TriGenSecondTriangle) {
                        b = a + side + 1;
                        c = a + 1;
                    }
                    else {
                        b = a + side;
                        c = a + side + 1;
                    }
                }

                // The filler layout only depends on s, so with other grid sizes it can use vertices that weren't loaded
                // here. The microcode would draw whatever is in them; RT64 may not have them in this workload.
                if ((a >= count) || (b >= count) || (c >= count)) {
                    RT64_LOG_PRINTF("F5Rogue: TRI_GEN filler uses vertices that weren't generated (n %u, s %u)", n, shift);
                    continue;
                }

                drawTriangle(state, a, b, c, a);
            }
        }

        static void triGen(State *state, DisplayList **dl) {
            TriGenCommand cmd;
            for (uint32_t i = 0; i < std::size(cmd.words); i++) {
                cmd.words[i] = (i & 1) ? (*dl)[i / 2].w1 : (*dl)[i / 2].w0;
            }

            *dl = *dl + (TriGenSize / sizeof(DisplayList)) - 1;

            // Widescreen: terrain is never anchored.
            setViewportAnchor(state, G_EX_ORIGIN_NONE);

            if (cmd.byte(2) & 0x2) {
                triGenQuad(state, cmd);
            }
            else {
                triGenGrid(state, cmd);
            }
        }

        // TEXRECT_GEN: a texture rectangle around a transformed vertex.
        static void texRectGen(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t primColor = (*dl)->w1;
            const uint32_t fogColor = (*dl)[1].w0;
            const uint32_t sizes = (*dl)[1].w1;
            const uint32_t extents = (*dl)[2].w0;
            *dl = *dl + 2;

            uint32_t index;
            if (!vertexIndex((w0 >> 8) & 0xFF, index)) {
                RT64_LOG_PRINTF("F5Rogue: invalid TEXRECT_GEN vertex 0x%08X", w0);
                return;
            }

            // Nothing is drawn outside of the guard band or beyond near or far.
            const Vertex &v = context.vertices[index];
            if (v.clipFlags & ClipRequiredMask) {
                return;
            }

            const bool flip = (w0 & 0x1) != 0;
            const bool mirrorS = (w0 & 0x2) != 0;
            const bool mirrorT = (w0 & 0x4) != 0;

            // Half extents in quarter pixels: size / w * projection scale * viewport scale / 2, with the projection's
            // [0][0] for x and [1][1] for y. With flip, the sizes are swapped first.
            int16_t size[2] = { int16_t(sizes >> 16), int16_t(sizes) };
            if (flip) {
                std::swap(size[0], size[1]);
            }

            const uint16_t perspNorm = dmemRead16(DmemPerspNorm);
            int16_t half[2];
            for (uint32_t j = 0; j < 2; j++) {
                const int32_t sizeOverW = mulFixed(clamp32(int64_t(size[j]) * v.invW), perspNorm);
                const int32_t viewportHalf = dmemReadS16(DmemViewportScale + j * 2) >> 1;
                const int32_t scaled = clamp32(int64_t(sizeOverW) * viewportHalf);
                half[j] = clamp16(mulFixed(scaled, dmemMatrixElement(DmemProjection, j, j)) >> 16);
            }

            int16_t ul[2], lr[2];
            for (uint32_t j = 0; j < 2; j++) {
                ul[j] = int16_t(v.screen[j] - half[j]);
                lr[j] = int16_t(v.screen[j] + half[j]);
            }

            // Texture steps in s5.10: extent (10.5) * 64 / half extent. With flip, s goes with the y extent.
            const int16_t extent[2] = { int16_t(extents >> 16), int16_t(extents) };
            int16_t stepHalf[2] = { half[0], half[1] };
            if (flip) {
                std::swap(stepHalf[0], stepHalf[1]);
            }

            int32_t step[2];
            for (uint32_t j = 0; j < 2; j++) {
                const int32_t inverse = (stepHalf[j] == 0) ? INT32_MAX : clamp32((int64_t(1) << 31) / stepHalf[j]);
                step[j] = clamp16((int64_t(extent[j]) * mulFixed(inverse, 0x80)) >> 16);
            }

            // Mirroring starts at the far end and steps back.
            int32_t start[2] = { 0, 0 };
            if (mirrorS) {
                step[0] = -step[0];
                start[0] = uint16_t(extents >> 16);
            }

            if (mirrorT) {
                step[1] = -step[1];
                start[1] = uint16_t(extents);
            }

            // The prim depth and colors are sent even if the rectangle ends up empty.
            const int32_t screenZ = int32_t((uint32_t(uint16_t(v.screen[2])) << 16) | v.screenZFraction);
            const uint32_t primDepth = uint32_t(screenZ >> 11) & 0xFFFF;
            sendRDP(state, uint32_t(G_SETPRIMDEPTH & 0x3F) << 24, primDepth << 16);
            sendRDP(state, uint32_t(G_SETPRIMCOLOR & 0x3F) << 24, primColor);
            if (dmemRead32(DmemGeometryMode) & GeometryFog) {
                // The fog color's alpha is the vertex's fog value.
                sendRDP(state, uint32_t(G_SETFOGCOLOR & 0x3F) << 24, (fogColor & 0xFFFFFF00U) | (v.rgba & 0xFF));
            }

            // Clamped to the top left only. The start coordinates move by the part that was cut off.
            int32_t cut[2];
            for (uint32_t j = 0; j < 2; j++) {
                cut[j] = std::min<int32_t>(ul[j], 0);
                ul[j] = std::max<int16_t>(ul[j], 0);
                lr[j] = std::max<int16_t>(lr[j], 0);
            }

            if (flip) {
                std::swap(cut[0], cut[1]);
            }

            for (uint32_t j = 0; j < 2; j++) {
                const int32_t negatedStep = -(step[j] * 64);
                start[j] += clamp16((int64_t(negatedStep) * cut[j]) >> 16) * 8;
            }

            if (((lr[0] - ul[0]) <= 0) || ((lr[1] - ul[1]) <= 0)) {
                return;
            }

            // The start coordinates are moved back by half a texel.
            const uint32_t command = (G_TEXRECT & 0x3F) | (flip ? 0x1 : 0x0);
            const uint32_t tile = dmemRead8(DmemTexture + 2) & 0x7;
            const uint32_t rectW0 = (command << 24) | (((uint32_t(uint16_t(lr[0])) << 12) | uint16_t(lr[1])) & 0xFFFFFF);
            const uint32_t rectW1 = (tile << 24) | (((uint32_t(uint16_t(ul[0])) << 12) | uint16_t(ul[1])) & 0xFFFFFF);
            const uint32_t rectW2 = (uint32_t(start[0] - 0x10) << 16) | (uint32_t(start[1] - 0x10) & 0xFFFF);
            const uint32_t rectW3 = (uint32_t(step[0]) << 16) | (uint32_t(step[1]) & 0xFFFF);
            drawTexRect(state, rectW0, rectW1, rectW2, rectW3, false);
        }

        // Other commands.

        static void noOp(State *state, DisplayList **dl) {
            // Do nothing.
        }

        static void matrix(State *state, DisplayList **dl) {
            const uint32_t flags = (*dl)->p0(16, 8);
            const uint32_t address = (*dl)->w1 & RSP_DMA_MASK;
            RSP *rsp = state->rsp.get();
            if (flags & 0x1) {
                dmaRead(state, DmemProjection, address, 0x3F);
                rsp->matrixCommon(negateY(dmemMatrix(DmemProjection)), address, uint8_t(rsp->projMask | rsp->loadMask));

                // The combined matrix isn't updated, so the previous one stays in use until the next modelview.
                rsp->modelViewProjMatrix = negateY(dmemMatrix(DmemCombined));
                rsp->modelViewProjInserted = true;
                rsp->modelViewProjChanged = false;
            }
            else {
                dmaRead(state, DmemModelView, address, 0x3F);

                // Combined = modelview x projection. There's no matrix stack and no multiplication with the previous modelview.
                for (uint32_t i = 0; i < 4; i++) {
                    for (uint32_t j = 0; j < 4; j++) {
                        int64_t sum = 0;
                        for (uint32_t k = 0; k < 4; k++) {
                            sum += int64_t(dmemMatrixElement(DmemModelView, i, k)) * dmemMatrixElement(DmemProjection, k, j);
                        }

                        const uint32_t value = uint32_t(clamp32(sum >> 16));
                        const uint32_t offset = (i * 4 + j) * 2;
                        dmemWrite16(DmemCombined + offset, uint16_t(value >> 16));
                        dmemWrite16(DmemCombined + MatrixFractionOffset + offset, uint16_t(value));
                    }
                }

                rsp->matrixCommon(dmemMatrix(DmemModelView), address, uint8_t(rsp->loadMask));
            }
        }

        static void vertexColor(State *state, DisplayList **dl) {
            dmaRead(state, DmemColorTable, (*dl)->w1, (*dl)->w0 & 0xFFFF);
        }

        // Widescreen: a viewport from the game replaces the HUD anchor (it is set again by the next HUD triangle).
        static void dropViewportAnchorOnWrite(uint32_t address, uint32_t length) {
            address &= DmemMask;
            if ((address < (DmemViewportTranslate + 8)) && ((address + length) > DmemViewportScale)) {
                context.viewportAnchor = G_EX_ORIGIN_NONE;
            }
        }

        static void moveMem(State *state, DisplayList **dl) {
            // The next 16 bytes go to the address in the table (0x80 is the viewport, 0x82 the texture coordinate scale).
            const uint32_t address = dmemRead16(DmemMoveMemTable + (*dl)->p0(16, 8));
            const uint32_t words[4] = { (*dl)[1].w0, (*dl)[1].w1, (*dl)[2].w0, (*dl)[2].w1 };
            *dl = *dl + 2;

            // SQV only stores the bytes up to the end of the 16-byte line.
            const uint32_t length = 16 - (address & 0xF);
            for (uint32_t i = 0; i < length; i++) {
                dmemWrite8(address + i, uint8_t(words[i / 4] >> (24 - (i % 4) * 8)));
            }

            dropViewportAnchorOnWrite(address, length);
            syncViewport(state);
        }

        static void moveWord(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t address = dmemRead16(DmemMoveWordTable + (w0 & 0xFF)) + ((w0 >> 8) & 0xFFFF);
            dmemWrite32(address, (*dl)->w1);
            dropViewportAnchorOnWrite(address, 4);
            syncViewport(state);
        }

        static void vertex(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t address = (*dl)->w1;
            dmaRead(state, DmemVertexInput, address, w0 & 0x3FF);
            const uint32_t count = transformVertices((w0 >> 10) & 0x3F);
            loadVertices(state, address & RSP_DMA_MASK, count);
        }

        static void tri1(State *state, DisplayList **dl) {
            triangleCommand(state, dl, false);
        }

        static void tri2(State *state, DisplayList **dl) {
            triangleCommand(state, dl, true);
        }

        static void runDl(State *state, DisplayList **dl) {
            // No depth limit is checked.
            const uint32_t readOffset = uint32_t(reinterpret_cast<uint8_t *>(*dl + 1) - state->fromRDRAM(context.blockAddress));
            context.callStack.push_back({ context.blockAddress, readOffset });
            fetchBlock(state, dl, (*dl)->w1);
        }

        static void branchDl(State *state, DisplayList **dl) {
            fetchBlock(state, dl, (*dl)->w1);
        }

        static void endDl(State *state, DisplayList **dl) {
            // The task ends when the stack is empty.
            if (context.callStack.empty()) {
                *dl = nullptr;
                return;
            }

            // The block is fetched again and runs from the saved read pointer.
            const CallStackEntry entry = context.callStack.back();
            context.callStack.pop_back();
            fetchBlock(state, dl, entry.blockAddress);
            *dl = reinterpret_cast<DisplayList *>(state->fromRDRAM(context.blockAddress) + entry.readOffset) - 1;
        }

        static void nextBlock(State *state, DisplayList **dl) {
            fetchBlock(state, dl, context.nextBlockAddress);
        }

        static void texture(State *state, DisplayList **dl) {
            // Only w0 is kept. Its bit 0 turns geometry mode bit 1 (texturing) on or off.
            const uint32_t w0 = (*dl)->w0;
            dmemWrite32(DmemTexture, w0);
            dmemWrite16(DmemGeometryMode + 2, uint16_t((dmemRead16(DmemGeometryMode + 2) & 0xFFFD) | ((w0 & 0x1) << 1)));
        }

        static void setGeometryMode(State *state, DisplayList **dl) {
            dmemWrite32(DmemGeometryMode, dmemRead32(DmemGeometryMode) | (*dl)->w1);
        }

        static void clearGeometryMode(State *state, DisplayList **dl) {
            dmemWrite32(DmemGeometryMode, dmemRead32(DmemGeometryMode) & ~(*dl)->w1);
        }

        // The other mode words are kept as an RDP SetOtherMode command and sent after every change.
        static void setOtherModeCommon(State *state, uint32_t address, uint32_t keepMask, uint32_t value) {
            dmemWrite32(address, (dmemRead32(address) & keepMask) | value);
            sendRDP(state, dmemRead32(DmemOtherModeH), dmemRead32(DmemOtherModeL));
        }

        static uint32_t otherModeKeepMask(uint32_t w0) {
            const uint32_t shift = (w0 >> 8) & 0xFF;
            const uint32_t length = w0 & 0xFF;
            return ~(((1U << (length & 31)) - 1) << (shift & 31));
        }

        static void setOtherModeH(State *state, DisplayList **dl) {
            setOtherModeCommon(state, DmemOtherModeH, otherModeKeepMask((*dl)->w0), (*dl)->w1);
        }

        static void setOtherModeL(State *state, DisplayList **dl) {
            setOtherModeCommon(state, DmemOtherModeL, otherModeKeepMask((*dl)->w0), (*dl)->w1);
        }

        // The mask of the bits to keep is the next word.
        static void setOtherModeHEx(State *state, DisplayList **dl) {
            const uint32_t keepMask = (*dl)[1].w0;
            const uint32_t value = (*dl)->w1;
            *dl = *dl + 1;
            setOtherModeCommon(state, DmemOtherModeH, keepMask, value);
        }

        static void setOtherModeLEx(State *state, DisplayList **dl) {
            const uint32_t keepMask = (*dl)[1].w0;
            const uint32_t value = (*dl)->w1;
            *dl = *dl + 1;
            setOtherModeCommon(state, DmemOtherModeL, keepMask, value);
        }

        // Command dispatch.

        // The microcode's handler table (DMEM 0x0D6).
        static const GBIFunction CommandTable[] = {
            &noOp, &matrix, &vertexColor, &moveMem, &vertex, &triGen, &runDl, &branchDl, &tri1, &setOtherModeHEx, &texRectGen,
            &moveWord, &texture, &setOtherModeH, &setOtherModeL, &endDl, &setGeometryMode, &clearGeometryMode, &nextBlock, &tri2,
            &setOtherModeLEx
        };

        static void spCommand(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            uint32_t entry = UINT32_MAX;
            if ((w0 >> 30) == 0) {
                // Opcodes 0x00-0x3F are the table index.
                entry = w0 >> 24;
            }
            else {
                // Opcodes 0x40-0xBF index the table from its end with -(w0 >> 23) as a byte offset from DMEM 0x064, so
                // bit 23 is part of the key: 0xBF is entry 8 and 0xBF with bit 23 is entry 7.
                const uint32_t offset = uint32_t(-(int32_t(w0) >> 23)) & 0x1FE;
                const uint32_t tableOffset = 0x0D6 - 0x064;
                if (offset >= tableOffset) {
                    entry = (offset - tableOffset) / 2;
                }
            }

            if (entry < std::size(CommandTable)) {
                CommandTable[entry](state, dl);
            }
            else {
                RT64_LOG_PRINTF("F5Rogue: unknown command 0x%08X 0x%08X", w0, (*dl)->w1);
            }

            advanceBlock(state, dl);
        }

        // Opcodes 0xC0-0xFF go to the RDP. Texture rectangles take the next 8 bytes too.
        static void rdpCommand(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            if ((w0 >> 25) == (G_TEXRECT >> 1)) {
                drawTexRect(state, w0, (*dl)->w1, (*dl)[1].w0, (*dl)[1].w1, true);
                *dl = *dl + 1;
            }
            else {
                sendRDP(state, w0, (*dl)->w1);
            }

            advanceBlock(state, dl);
        }

        static void reset(State *state) {
            // The task copies 0x800 bytes from the data pointer to DMEM: the microcode's constants in the first 0x100
            // bytes and whatever follows them in RDRAM. The state is then initialized.
            dmaRead(state, 0, state->ext.interpreter->UCode.dataAddress, DmemTaskDataSize - 1);
            dmemWrite32(DmemGeometryMode, 0);
            dmemWrite32(DmemOtherModeH, uint32_t(G_RDPSETOTHERMODE & 0x3F) << 24);
            dmemWrite32(DmemOtherModeL, 0);
            dmemWrite16(DmemFogScale, 0xFF);
            context.callStack.clear();
            context.blockEnd = nullptr;
            context.blockFetched = false;
            context.viewportSynced = false;
            context.viewportAnchor = G_EX_ORIGIN_NONE;
            context.lastLoadCount = 0;
            context.hudRegionsPrevious.swap(context.hudRegions);
            context.hudRegions.clear();

            // Full-width texture rectangles in this game are 2D art (title screen, menu photos): keep them 4:3 and
            // centered with the Expand aspect ratio instead of stretching them. Framebuffer copies and fill rectangle
            // clears still stretch.
            state->rdp->extended.global.adjustFullWidthRects = true;
        }

        void setup(GBI *gbi) {
            // RT64 never sees the microcode's bits 12 and 13 (see the top of this file), so these culling masks never match.
            gbi->constants = {
                { F3DENUM::G_MTX_MODELVIEW, 0x00 },
                { F3DENUM::G_MTX_PROJECTION, 0x01 },
                { F3DENUM::G_MTX_MUL, 0x00 },
                { F3DENUM::G_MTX_LOAD, 0x02 },
                { F3DENUM::G_MTX_NOPUSH, 0x00 },
                { F3DENUM::G_MTX_PUSH, 0x04 },
                { F3DENUM::G_TEXTURE_ENABLE, GeometryTexture },
                { F3DENUM::G_SHADING_SMOOTH, GeometrySmooth },
                { F3DENUM::G_CULL_FRONT, 0x00001000 },
                { F3DENUM::G_CULL_BACK, 0x00002000 },
                { F3DENUM::G_CULL_BOTH, 0x00003000 }
            };

            // The common RDP functions are already in the map.
            for (uint32_t i = 0; i < context.rdpFunctions.size(); i++) {
                context.rdpFunctions[i] = gbi->map[0xC0 | i];
            }

            context.rdpFunctions[G_NOOP] = &GBI_RDP::noOp;

            for (uint32_t i = 0; i < 0xC0; i++) {
                gbi->map[i] = &spCommand;
            }

            for (uint32_t i = 0xC0; i < UCODE_MAP_SIZE; i++) {
                gbi->map[i] = &rdpCommand;
            }

            gbi->resetFromTask = &reset;
            gbi->displayListStart = &displayListStart;
        }
    }
};
