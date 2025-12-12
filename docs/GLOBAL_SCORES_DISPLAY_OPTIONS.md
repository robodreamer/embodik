# Alternative Global Scores Display Options

This document outlines different ways to display global scores beyond the current text-based format.

## Current Implementation
- **Format**: Multi-line text display
- **Features**: Dropdowns for metric selection and display mode
- **Shows**: Mean, Min, Max, Std, Weighted Avg, Coverage Weighted, Count

---

## Option 1: Progress Bar Style (Visual Bars)
**Concept**: Use sliders or custom visual bars to represent normalized scores

**Pros**:
- Highly visual and intuitive
- Easy to compare metrics at a glance
- Works well with Viser's slider components

**Implementation**:
- Use read-only sliders positioned at normalized values (0-1)
- Color-code: Green (>0.7), Yellow (0.4-0.7), Red (<0.4)
- Show metric name + value above each bar

**Example Layout**:
```
Manipulability: 0.067 [████░░░░░░] 6.7%
RangeOfMotion:  0.894 [██████████] 89.4%
Singularity:    0.455 [██████░░░░] 45.5%
```

---

## Option 2: Color-Coded Text Display
**Concept**: Enhance current text with color coding and better formatting

**Pros**:
- Minimal changes to current implementation
- Color provides quick visual feedback
- Maintains detailed information

**Implementation**:
- Use Unicode box-drawing characters for structure
- Color-code values: Green (good), Yellow (medium), Red (poor)
- Use emoji indicators: ✅ 🟡 ⚠️

**Example Layout**:
```
┌─────────────────────────────────┐
│ Manipulability: 0.067 🟡        │
│   Mean: 0.067113                │
│   Range: [0.030, 0.097]         │
│   Count: 180                     │
├─────────────────────────────────┤
│ RangeOfMotion: 0.894 ✅         │
│   Mean: 0.893561                │
│   Range: [0.344, 0.992]          │
│   Count: 180                     │
└─────────────────────────────────┘
```

---

## Option 3: Compact Card Layout
**Concept**: Individual "cards" for each metric with key stats

**Pros**:
- Clean, organized appearance
- Easy to scan
- Can show multiple metrics simultaneously

**Implementation**:
- Each metric gets its own section
- Large number for primary value
- Smaller text for details
- Color-coded background or border

**Example Layout**:
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Manipulab.  │  │ Range Motion│  │ Singularity │
│    0.067    │  │    0.894    │  │    0.455    │
│ Mean: 0.067 │  │ Mean: 0.894 │  │ Mean: 0.455 │
│ [0.030-0.097]│ │ [0.344-0.992]│ │ [0.003-0.757]│
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## Option 4: Gauge/Meter Style
**Concept**: Circular or linear gauges showing score as percentage

**Pros**:
- Very visual representation
- Familiar gauge/meter metaphor
- Good for quick assessment

**Implementation**:
- Use sliders positioned at score values (read-only)
- Add text labels showing percentage
- Use color gradients

**Example Layout**:
```
Manipulability:  [░░░░░░░░░░] 6.7%
RangeOfMotion:   [██████████] 89.4%
Singularity:     [██████░░░░] 45.5%
```

---

## Option 5: Comparison Table Format
**Concept**: Structured table with columns for different statistics

**Pros**:
- Easy to compare across metrics
- Professional appearance
- All data visible at once

**Implementation**:
- Use monospace font for alignment
- Column headers: Metric | Mean | Min | Max | Std | Count
- Color-code rows based on mean value

**Example Layout**:
```
Metric            Mean      Min       Max       Std       Count
─────────────────────────────────────────────────────────────────
Manipulability    0.067     0.030     0.097     0.015     180
RangeOfMotion     0.894     0.344     0.992     0.089     180
Singularity       0.455     0.003     0.757     0.142     180
```

---

## Option 6: Sparkline-Style Mini Charts
**Concept**: ASCII art mini charts showing distribution

**Pros**:
- Shows distribution, not just summary stats
- Unique visual representation
- Compact

**Implementation**:
- Use ASCII characters to create simple bar charts
- Show min, mean, max positions
- Requires distribution data (could use histogram)

**Example Layout**:
```
Manipulability: 0.067
  Min ──●───────────────●── Max
       0.030    0.067   0.097

RangeOfMotion: 0.894
  Min ────────────●───────● Max
       0.344      0.894   0.992
```

---

## Option 7: Multi-Mode Display (Recommended)
**Concept**: Add a display style dropdown with multiple options

**Pros**:
- User can choose preferred format
- Flexible and extensible
- Best of all worlds

**Implementation**:
- Add "Display Style" dropdown: ["Text", "Bars", "Cards", "Table"]
- Implement each style as separate method
- Switch between styles based on selection

**Styles**:
1. **Text** (current): Detailed multi-line text
2. **Bars**: Visual progress bars
3. **Cards**: Compact card layout
4. **Table**: Structured table format

---

## Recommendation: Option 7 (Multi-Mode Display)

**Why**:
- Provides flexibility for different use cases
- Easy to implement incrementally
- Users can choose what works best for them
- Can add more styles later

**Implementation Plan**:
1. Add "Display Style" dropdown to GUI
2. Create separate rendering methods for each style
3. Update `_update_global_scores_display()` to route to selected style
4. Start with 2-3 styles, add more as needed

**Priority Styles**:
1. **Bars** (most visual impact)
2. **Table** (best for comparison)
3. **Cards** (best for overview)

---

## Technical Considerations

### Viser GUI Limitations
- No native charting widgets
- Text display is primary method
- Sliders can be used for visual bars (read-only)
- Unicode/ASCII art works well

### Implementation Notes
- Use Unicode box-drawing characters: ┌ ┐ └ ┘ │ ─ ├ ┤
- Use emoji for indicators: ✅ 🟡 ⚠️ 🔴 🟢
- Color coding via text formatting (if supported)
- Monospace font for alignment

### Performance
- All options are text-based, so performance is similar
- No significant overhead for any option
- Can cache formatted strings if needed

