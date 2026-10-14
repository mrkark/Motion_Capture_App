# Спецификация форматов JSON — Karate Motion Studio

Данный документ описывает структуру данных, системы координат и спецификацию экспорта форматов JSON в приложении **Karate Motion Studio** для интеграции с проектом **Bunkai Explorer** (React, Three.js).

---

## 1. Система координат

Все 3D-данные нормализованы под стандартную систему координат **Three.js** и метрическую систему:

| Ось | Направление в пространстве Three.js | Физический смысл |
|---|---|---|
| **+X** | Вправо от зрителя | Левая сторона каратиста (при взгляде спереди) |
| **+Y** | Вверх | Вертикальная ось (высота от татами) |
| **+Z** | Вперед (к камере) | Направление взгляда / движения вперед |
| **Единицы** | Метры (meters) | 1.0 = 1 метр |

MediaPipe World Landmarks по умолчанию возвращает координаты в системе (X вправо, Y вниз, Z от камеры вглубь). В `MocapTracker` применяется автоматическое преобразование:
```python
x_three = lm.x
y_three = -lm.y
z_three = -lm.z
```

---

## 2. Формат 1: Capture JSON (`bunkai-capture-json`)

Предназначен для хранения очищенных 3D-траекторий суставов, анализа дистанции (Ма-ай), углов ударов и визуализации скелета в Three.js с помощью `LineSegments` или `SphereGeometry`.

### Спецификация верхнего уровня

```json
{
  "format": "bunkai-capture-json",
  "version": 1,
  "technique": "gedan_barai",
  "name": "gedan_barai_sensei_01",
  "tags": ["shotokan", "gedan_barai", "block", "zenkutsu_dachi"],
  "fps": 30.0,
  "duration": 2.133,
  "frameCount": 64,
  "coordinateSystem": {
    "handedness": "right-handed",
    "upAxis": "Y",
    "forwardAxis": "Z",
    "units": "meters",
    "description": "Three.js compatible world coordinates: X right, Y up, Z forward"
  },
  "frames": [
    {
      "frame": 0,
      "time": 0.0,
      "rootPosition": [0.0, 0.9, 0.0],
      "joints": {
        "nose": [0.012, 1.62, 0.08],
        "left_shoulder": [-0.21, 1.41, 0.02],
        "right_shoulder": [0.21, 1.41, -0.01],
        "left_elbow": [-0.35, 1.15, 0.05],
        "right_elbow": [0.38, 1.12, -0.04],
        "left_wrist": [-0.28, 0.85, 0.18],
        "right_wrist": [0.25, 0.95, -0.15],
        "left_hip": [-0.14, 0.92, 0.0],
        "right_hip": [0.14, 0.92, 0.0],
        "left_knee": [-0.18, 0.52, 0.22],
        "right_knee": [0.22, 0.48, -0.35],
        "left_ankle": [-0.21, 0.12, 0.38],
        "right_ankle": [0.26, 0.11, -0.62]
      },
      "confidence": {
        "left_shoulder": 0.98,
        "right_shoulder": 0.97,
        "left_wrist": 0.91,
        "right_wrist": 0.89
      },
      "tracking_loss": false,
      "head": {
        "yaw": 0.0,
        "pitch": -0.02,
        "roll": 0.0
      },
      "fingers": {
        "left": {
          "thumb": 0.9,
          "index": 0.85,
          "middle": 0.9,
          "ring": 0.85,
          "pinky": 0.9,
          "detected": true
        },
        "right": {
          "thumb": 0.9,
          "index": 0.85,
          "middle": 0.9,
          "ring": 0.85,
          "pinky": 0.9,
          "detected": true
        }
      }
    }
  ]
}
```

### Набор суставов (`STANDARD_JOINTS`, 33 ключевые точки):
- Голова: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `mouth_left`, `mouth_right`.
- Плечевой пояс и руки: `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_pinky`, `right_pinky`, `left_index`, `right_index`, `left_thumb`, `right_thumb`.
- Таз и ноги: `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`.

---

## 3. Формат 2: Retargeted Animation JSON (`bunkai-retargeted-animation`)

Предназначен для **мгновенного применения анимации в браузере** к скелету 3D-модели каратиста (`Karateka_Armature`) без промежуточных конвертаций.

Каждая кость представлена **локальным кватернионом поворота** относительно родительской кости в пространстве рига:
`[x, y, z, w]`, где $\|Q\| = 1.0$.

### Спецификация

```json
{
  "format": "bunkai-retargeted-animation",
  "version": 1,
  "technique": "oi_zuki",
  "name": "oi_zuki_chudan",
  "tags": ["shotokan", "punch", "oi_zuki", "chudan"],
  "targetRig": "Karateka_Armature",
  "fps": 30.0,
  "duration": 1.966,
  "frameCount": 60,
  "frames": [
    {
      "frame": 0,
      "time": 0.0,
      "rootPosition": [0.0, 0.92, 0.0],
      "rootRotation": [0.0, 0.0, 0.0, 1.0],
      "bones": {
        "Hips": [0.0, 0.0, 0.0, 1.0],
        "Spine": [0.021, 0.0, 0.0, 0.999],
        "Chest": [0.021, 0.0, 0.0, 0.999],
        "Head": [0.0, 0.012, 0.0, 0.999],
        "UpperArm.L": [0.152, 0.384, -0.211, 0.884],
        "LowerArm.L": [0.082, -0.412, 0.121, 0.901],
        "UpperArm.R": [-0.141, -0.218, 0.091, 0.962],
        "LowerArm.R": [0.023, 0.541, -0.112, 0.833],
        "UpperLeg.L": [0.214, 0.022, 0.015, 0.976],
        "LowerLeg.L": [-0.342, 0.011, -0.008, 0.939],
        "UpperLeg.R": [-0.184, -0.015, -0.012, 0.982],
        "LowerLeg.R": [0.112, 0.005, 0.002, 0.993],
        "Thumb1.L": [0.654, 0.0, 0.0, 0.756],
        "Thumb2.L": [0.654, 0.0, 0.0, 0.756],
        "Index1.L": [0.812, 0.0, 0.0, 0.583],
        "Index2.L": [0.724, 0.0, 0.0, 0.689],
        "Index3.L": [0.724, 0.0, 0.0, 0.689]
      },
      "fingers": {
        "left": { "thumb": 0.9, "index": 0.85, "middle": 0.9, "ring": 0.85, "pinky": 0.9, "detected": true },
        "right": { "thumb": 0.9, "index": 0.85, "middle": 0.9, "ring": 0.85, "pinky": 0.9, "detected": true }
      }
    }
  ]
}
```

### Иерархия костей `Karateka_Armature`

```
Hips (Root Translation + Rotation)
├── Spine
│   └── Spine1
│       └── Chest
│           ├── Neck
│           │   └── Head
│           ├── Shoulder.L
│           │   └── UpperArm.L
│           │       └── LowerArm.L
│           │           └── Hand.L
│           │               ├── Thumb1.L .. Thumb3.L
│           │               ├── Index1.L .. Index3.L
│           │               ├── Middle1.L .. Middle3.L
│           │               ├── Ring1.L .. Ring3.L
│           │               └── Pinky1.L .. Pinky3.L
│           └── Shoulder.R
│               └── UpperArm.R
│                   └── LowerArm.R
│                       └── Hand.R
│                           ├── Thumb1.R .. Thumb3.R
│                           ├── Index1.R .. Index3.R
│                           ├── Middle1.R .. Middle3.R
│                           ├── Ring1.R .. Ring3.R
│                           └── Pinky1.R .. Pinky3.R
├── UpperLeg.L
│   └── LowerLeg.L
│       └── Foot.L
└── UpperLeg.R
    └── LowerLeg.R
        └── Foot.R
```

---

## 4. Формат 3: Проектный файл (`karate-motion-studio`)

Сохраняет полный рабочий процесс студии:
- Исходные данные захвата (`rawFrames`)
- Отфильтрованные траектории после 1€-фильтра (`filteredFrames`)
- Расчетные кватернионы костей (`retargeting`)
- Диапазон обрезки таймлайна (`trimRange`)
- Метаданные техники, частоту кадров и имя источника.

Позволяет повторно открыть захваченное движение, настроить параметры фильтрации или обрезки без повторного прогона видео через MediaPipe.
