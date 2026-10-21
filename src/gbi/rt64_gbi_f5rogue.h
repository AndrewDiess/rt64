//
// RT64
//
// Copyright (c) 2026 RT64 Contributors
// SPDX-License-Identifier: MIT
//

#pragma once

#include "rt64_gbi.h"

// Custom graphics microcode used by Star Wars: Rogue Squadron (Factor 5).
//
// Display lists are chains of blocks with an 8-byte header and use raw RDRAM addresses. Vertices are
// positions only: colors and texture coordinates are assigned per triangle. There are also commands
// that generate billboards (TEXRECT_GEN) and terrain patches (TRI_GEN) on the RSP.

namespace RT64 {
    namespace GBI_F5ROGUE {
        void setup(GBI *gbi);
    };
};
