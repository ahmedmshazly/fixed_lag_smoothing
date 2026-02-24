# Umbrella World Fixed-Lag Smoothing Engine for Hidden Markov Models


A modular, mathematically stable, object-oriented Python engine for performing Fixed-Lag Smoothing in a Hidden Markov Model (HMM) environment. 

This project implements a robust algorithmic solution to the classic Umbrella World problem (from Russell & Norvig's Artificial Intelligence: A Modern Approach). It demonstrates how to update past beliefs based on future evidence while strictly preventing floating-point underflow (catastrophic cancellation) during extended time-series processing. 

As a bonus, it includes a fully-featured, interactive Graphical User Interface (GUI) built on an MVC architecture to visualize the smoothing process in real-time.

---

## 1. Project Overview

In real-world tracking, sensors are flawed, and the underlying state of the world is hidden. 

* Filtering estimates the present state based on evidence up to the present. 
* Smoothing goes a step further: it delays the final estimation by a lag of L days, using the evidence from those L future days to mathematically correct and refine our belief about the past.

This codebase is designed with a strict separation of concerns. It decouples the World (domain rules like weather and sensors) from the Engine (the mathematical operations handling forward filtering and backward smoothing), making it highly extensible to other domains like robot localization or stock market prediction.

---

## 2. Academic & Mathematical Foundation



### The Umbrella World Concept

We model a world with two hidden states: Rain and Sun. We cannot observe the weather directly; we only observe whether the director brings an umbrella or not. 

```mermaid
graph TD
    %% ==========================================
    %% UMBRELLA WORLD: HIDDEN MARKOV MODEL (HMM)
    %% ==========================================

    %% --- 1. HIDDEN STATES (The underlying truth) ---
    subgraph Hidden_States ["🌤️ True Weather (Hidden States)"]
        direction LR
        Rain((🌧️ Rain))
        Sun((☀️ Sun))
    end

    %% --- 2. OBSERVATIONS (The visible evidence) ---
    subgraph Observations ["🌂 Director's Action (Observations)"]
        direction LR
        Umbrella{{🌂 Umbrella Seen}}
        NoUmbrella{{🚫 No Umbrella}}
    end

    %% ==========================================
    %% --- 3. TRANSITION MODEL (T Matrix) ---
    %% ==========================================
    
    %% Self-transitions (Weather persistence)
    Rain -->|"0.7 (Stays Rainy)"| Rain
    Sun -->|"0.7 (Stays Sunny)"| Sun
    
    %% Cross-transitions (Weather changing)
    Rain -->|"0.3 (Clears up)"| Sun
    Sun -->|"0.3 (Starts Raining)"| Rain

    %% ==========================================
    %% --- 4. SENSOR MODEL (O Matrix) ---
    %% ==========================================
    
    %% If it is RAINING today (Row 1 of O Matrix):
    Rain -.->|"0.9 (Brings Umbrella)"| Umbrella
    Rain -.->|"0.1 (Forgets Umbrella)"| NoUmbrella
    
    %% If it is SUNNY today (Row 2 of O Matrix):
    Sun -.->|"0.2 (Uses as Parasol)"| Umbrella
    Sun -.->|"0.8 (Leaves at Home)"| NoUmbrella

    %% ==========================================
    %% --- 5. STYLING & CLASSES ---
    %% ==========================================
    
    classDef stateNode fill:#4fc3f7,stroke:#01579b,stroke-width:3px,color:black,font-weight:bold,font-size:16px
    classDef obsNode fill:#ffe082,stroke:#ff8f00,stroke-width:3px,color:black,font-weight:bold,font-size:14px
    classDef groupBox fill:#f8f9fa,stroke:#b0bec5,stroke-width:2px,stroke-dasharray: 5 5,color:#37474f,font-weight:bold
    
    class Rain,Sun stateNode
    class Umbrella,NoUmbrella obsNode
    class Hidden_States,Observations groupBox
```

---

## 3. System Architecture (Technical Details)

The system relies on strict Object-Oriented principles, keeping the domain models cleanly separated from the mathematical engines.

### Package Dependencies (Component Architecture)

The execution scripts act as assembly lines. The world package knows nothing about the algorithms operating on it.

```mermaid
flowchart TD
    %% Styling classes defined first
    classDef main_file fill:#ffcc80,stroke:#e65100,stroke-width:3px,color:black,font-weight:bold
    classDef fileNode fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:black,font-weight:bold
    classDef ext fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,stroke-dasharray: 5 5,color:black

    %% Define standard nodes
    Main([main.py<br>Execution Entry Point])

    subgraph EnginePackage [📦 Package: engine]
        direction TB
        Smoother[smoother.py]
        Forward[forward.py]
        Backward[backward.py]
        Window[window.py]

        Smoother -->|Imports & Orchestrates| Window
        Smoother -->|Imports & Delegates| Forward
        Smoother -->|Imports & Delegates| Backward
    end

    subgraph WorldPackage [📦 Package: world]
        direction TB
        HMM[hmm.py]
        Transition[transition.py]
        Sensor[sensor.py]

        HMM -->|Groups| Transition
        HMM -->|Groups| Sensor
    end

    Numpy[(NumPy Library)]

    %% Cross-package dependencies
    Main -->|Instantiates| HMM
    Main -->|Instantiates| Smoother
    Smoother -->|References| HMM

    %% Math dependencies
    Forward -.->|Relies on| Numpy
    Backward -.->|Relies on| Numpy
    Transition -.->|Relies on| Numpy
    Sensor -.->|Relies on| Numpy

    %% Apply Styles down here to avoid parser errors
    class Main main_file
    class Smoother,Forward,Backward,Window,HMM,Transition,Sensor fileNode
    class Numpy ext
```

### Object-Oriented System Design

The FixedLagSmoother acts as an orchestrator. Instead of being a monolithic mathematical script, it delegates forward math to ForwardFilter, backward math to BackwardTransformer, and memory management to the EvidenceWindow.

```mermaid
classDiagram
    %% Package: Engine (The Mechanics)
    namespace engine {
        class FixedLagSmoother {
            - int lag
            - int t
            - deque f_history
            - ndarray T
            - ndarray T_transposed
            + __init__(hmm: HiddenMarkovModel, lag: int)
            - _normalize(vector: ndarray) ndarray
            - _format_prob(vector: ndarray) str
            + process_day(evidence: Any) ndarray
        }

        class BackwardTransformer {
            - int num_states
            - ndarray T
            + __init__(num_states: int, transition_matrix: ndarray)
            + compute_backward_message(window_evidence: List, sensor_model: Any) ndarray
        }

        class ForwardFilter {
            - ndarray f
            + __init__(prior: ndarray)
            + step_forward(T_transposed: ndarray, O_t: ndarray) ndarray
            + get_current_f() ndarray
        }

        class EvidenceWindow {
            - int lag
            - int max_size
            - deque queue
            + __init__(lag: int)
            + add(evidence: Any)
            + is_full() bool
            + pop_oldest() Any
            + get_oldest() Any
            + get_contents() list
        }
    }

    %% Package: World (The Domain)
    namespace world {
        class HiddenMarkovModel {
            - int num_states
            - ndarray prior
            + __init__(transition_model: TransitionModel, sensor_model: SensorModel, prior: List)
            + get_prior() ndarray
        }

        class SensorModel {
            - dict probs
            - int num_states
            + __init__(observation_probs: Dict)
            + get_O(evidence: Any) ndarray
        }

        class TransitionModel {
            - ndarray T
            - ndarray T_transposed
            + __init__(transition_matrix: list|ndarray)
            + get_T() ndarray
            + get_T_transposed() ndarray
        }
    }

    %% Relationships
    FixedLagSmoother *-- HiddenMarkovModel : references
    FixedLagSmoother *-- EvidenceWindow : orchestrates
    FixedLagSmoother *-- ForwardFilter : delegates forward math
    FixedLagSmoother *-- BackwardTransformer : delegates backward math
    
    HiddenMarkovModel *-- TransitionModel : contains
    HiddenMarkovModel *-- SensorModel : contains
```

### The Daily Processing Cycle

Every single day, without fail, the system updates its present-day belief. If the evidence window is full, it executes the backward pass to rectify the past.

```mermaid
flowchart TD
    %% Styling classes defined first
    classDef startEnd fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:white,font-weight:bold
    classDef processNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef decisionNode fill:#fff8e1,stroke:#ff8f00,stroke-width:2px,font-weight:bold
    classDef dataNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

    %% Define standard nodes and connections
    Start([Start: process_day]) --> ReceiveData[/Receive Daily Evidence/]

    subgraph Phase1 [PHASE 1: Present Timeline Filtering]
        direction TB
        AddEv[Add Evidence to Window]
        FetchO[Fetch Sensor Matrix O_t]
        StepFwd[ForwardFilter: Push Belief Forward]
        SaveF[Save f_t to f_history buffer]
        
        ReceiveData --> AddEv
        AddEv --> FetchO
        FetchO --> StepFwd
        StepFwd --> SaveF
    end

    subgraph Phase2_3 [PHASE 2 & 3: Memory & Stable Smoothing]
        direction TB
        IsFull{Is t >= lag + 1?}
        CalcB[BackwardTransformer: Compute 'b' over window]
        FetchOldF[Fetch old belief f_t-d from history]
        Multiply[Combine: f_t-d * b]
        Normalize[Normalize result to 1.0]
        SlideWindow[Slide Window: pop_oldest]
        SetRes[Result = Smoothed Belief]
        SetNone[Result = None]
        
        SaveF --> IsFull
        IsFull -- "Yes (Window Full)" --> CalcB
        CalcB --> FetchOldF
        FetchOldF --> Multiply
        Multiply --> Normalize
        Normalize --> SlideWindow
        SlideWindow --> SetRes
        
        IsFull -- "No (Window still filling)" --> SetNone
    end

    IncT[Increment Day: t = t + 1]
    ReturnResult([Return Result])
    
    SetRes --> IncT
    SetNone --> IncT
    IncT --> ReturnResult

    %% Apply Styles down here to avoid parser errors
    class Start,ReturnResult startEnd
    class AddEv,FetchO,StepFwd,SaveF,CalcB,FetchOldF,Multiply,Normalize,SlideWindow,IncT processNode
    class IsFull decisionNode
    class ReceiveData,SetRes,SetNone dataNode

```

```mermaid
sequenceDiagram
    autonumber
    actor Client as Main Script
    participant Smoother as FixedLagSmoother
    participant Window as EvidenceWindow
    participant Sensor as SensorModel
    participant Forward as ForwardFilter
    participant Backward as BackwardTransformer

    Client->>Smoother: process_day(evidence)
    
    rect rgb(240, 248, 255)
    Note over Smoother,Forward: PHASE 1: THE PRESENT (Filtering)
    Smoother->>Window: add(evidence)
    Smoother->>Sensor: get_O(evidence)
    Sensor-->>Smoother: O_t (Observation Matrix)
    
    Smoother->>Forward: step_forward(T_transposed, O_t)
    Smoother->>Forward: get_current_f()
    Forward-->>Smoother: f_t (Present Day Belief)
    Smoother->>Smoother: f_history.append(f_t)
    end
    
    rect rgb(245, 245, 245)
    Note over Smoother,Window: PHASE 2: MEMORY STATE
    Smoother->>Window: get_contents()
    Window-->>Smoother: window_evidence (List)
    end
    
    alt t >= lag + 1 (Window is Full enough to Smooth)
        rect rgb(255, 240, 245)
        Note over Smoother,Backward: PHASE 3: STABLE SMOOTHING
        Smoother->>Backward: compute_backward_message(window_evidence, sensor_model)
        Backward-->>Smoother: b (Backward Context Vector)
        
        Smoother->>Smoother: f_t_minus_d = f_history[0]
        Smoother->>Smoother: unnormalized = f_t_minus_d * b
        Smoother->>Smoother: result = _normalize(unnormalized)
        
        Smoother->>Window: pop_oldest() (Slide window forward)
        end
    else Window Not Full
        Smoother->>Smoother: result = None
    end
    
    Smoother-->>Client: return result
```

---

## 4. Handling Numerical Stability (Practical Details)

A common pitfall in standard HMM textbook implementations is catastrophic cancellation (floating-point underflow). When multiplying probabilities over long time horizons, the numbers quickly collapse to 0.0, resulting in division-by-zero errors.

In our engine (specifically the backward and forward steps), we strictly prevent underflow by normalizing the vectors at every individual time step. By dividing the raw probabilities by their sum at each step, we ensure the algorithm can run infinitely (e.g., a 10,000-day stress test) without mathematical failure, ensuring real-world practical application.

```python
# The stable backward formula
b_raw = self.T @ O_k @ b

# Normalize 'b' at each step to strictly prevent floating-point explosion or collapse
total = np.sum(b_raw)
if total > 0:
    b = b_raw / total
```

---

## 5. Installation & Usage

### Prerequisites

* Python 3.8+
* numpy (for mathematical matrix operations)
* matplotlib (for GUI plotting)

### Installation

Clone the repository and install the requirements using standard pip install commands:
python -m pip install numpy matplotlib

### Running the CLI Simulation (Stress Test)

To run the automated 35-day stress test script natively in the terminal, execute:

    python main.py

This runs a rigorous predefined sequence proving the engine's mathematical stability over extended timeframes, printing the step-by-step logic directly to the console.

---

## 6. Bonus: The Interactive GUI Module

While the engine is robust and strictly numerical, this repository includes a fully-featured Graphical User Interface (GUI) module, unlike the rest of modules, **built with LLMs help for better experience.** If you want to experience Fixed-Lag Smoothing in a user-friendly UI instead of reading terminal logs, running the GUI application is the way to go.

### How to Run the GUI

To launch the interactive GUI, simply run:

    python app.py

### High-Level Structure and How it Works

The GUI is built using Tkinter and Matplotlib, structured strictly around the Model-View-Controller (MVC) architectural pattern to ensure it does not interfere with the core mathematical engine.

* The Model: The core mathematical Engine and World packages you interact with via the CLI.
* The View: Composed of main_window.py (layout and user inputs), plotter.py (real-time Matplotlib charts), and math_view.py (mathematical formula rendering).
* The Controller: The simulation_controller.py acts as the brain of the GUI. 

When you start the simulation, the Controller initializes the FixedLagSmoother. As you feed evidence (Umbrella / No Umbrella) via the UI, the Controller passes this to the Engine. It then intercepts the calculated forward probabilities and smoothed backward probabilities, formats them, and broadcasts them to the Plotter and Log panels simultaneously. This allows you to visually watch the Forward Filter react in real-time, while the Smoothed Line trails L days behind, rectifying the past dynamically.

```

```
