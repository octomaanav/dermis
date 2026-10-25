"""The specialist agents. Each one is a Profile: same touch data, read through one application's needs.

Numbers here are engineering defaults to tune, not standards. Object yield forces come from the
dataset README (can 18 N, egg 25 N, cup 4 N, engineering estimates to be measured on the bench).
"""


from dataclasses import dataclass, field

# Fracture yield per object, N. Objects without an entry don't break in the sim.
OBJECT_YIELD_N = {"can": 18.0, "egg": 25.0, "cup": 4.0}

PODS = ("index_tip", "thumb_tip", "middle_tip", "thenar_apb")
HAND_PODS = {"index_tip": "index tip", "thumb_tip": "thumb tip", "middle_tip": "middle tip",
             "thenar_apb": "thenar (palm)"}

# Default advice per finding code. Profiles override the ones their domain handles differently.
DEFAULT_ACTIONS = {
    "force_over_limit": "Cap commanded grip force below the limit (enable SafeGrip) and re-run the same scenario.",
    "fault_fracture": "The object broke: lower the force ceiling for this object class and slow the close.",
    "fault_slip": "Object slipped: raise normal force toward the friction requirement, or regrasp with more contact area.",
    "fault_impact": "Object landed too fast: lower the place height and slow the release.",
    "fault_early_release": "Grip opened before the place: gate release on contact state, not on a timer.",
    "fault_overpress": "Sustained over-pressure: add a force-hold limit, not just a peak limit.",
    "low_margin": "Friction margin is thin: add normal force or use a higher-friction contact surface.",
    "slip_signal": "Slip-band vibration detected: use it as an early warning to tighten before the object moves.",
    "rail": "Magnetometer hit its rail: readings in that window are unreliable. Reduce gain or magnet strength, don't trust force there.",
    "noisy": "Sensor noise is above spec: check magnet mounting, cabling and nearby ferrous parts.",
    "frame_drops": "Frames were dropped: check the BLE/serial link and host load before trusting timing metrics.",
    "late_detection": "Touch was detected late from the field: lower the contact threshold or recalibrate the rest field.",
    "no_contact": "No contact was detected: check pod placement and that the grasp reached the object.",
}


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    title: str
    domain: str
    priorities: tuple[str, ...]
    keywords: tuple[str, ...]
    examples: tuple[str, ...]
    objects: tuple[str, ...] = ()           # dataset objects relevant to this domain; () = all
    applications: tuple[str, ...] = ("hand",)  # SpacetimeDB `application` tags this agent reads natively
    yield_safety: float = 0.8               # force limit = yield x this
    force_limit_n: float = 40.0             # ceiling for objects with no known yield
    min_margin: float = 0.15                # friction margin floor
    critical_faults: tuple[str, ...] = ("fracture", "slip", "impact")
    pod_labels: dict = field(default_factory=lambda: dict(HAND_PODS))
    actions: dict = field(default_factory=dict)
    proxy_note: str = ""                    # set when the data is a stand-in for this domain
    fleet_first: bool = False               # answer with a fleet view when no episode is named
    # Which checks this domain runs, most important first. Order decides the headline finding.
    checks: tuple[str, ...] = ("fault", "force", "margin", "slip_signal", "sensor", "timing")
    latency_ms: float = 30.0                # touch-detection lag this domain tolerates
    in_matrix: bool = True                  # judges single recordings in the cross-application compare
    kind: str = "touch"                     # "touch" specialist, or the general "explorer" / "visualizer"

    def action(self, code: str) -> str:
        return self.actions.get(code) or DEFAULT_ACTIONS.get(code, "")

    def limit_for(self, obj: str) -> float:
        y = OBJECT_YIELD_N.get(obj)
        return round(y * self.yield_safety, 2) if y else self.force_limit_n


PROFILES: list[Profile] = [
    Profile(
        key="humanoid_arm",
        name="dermis_humanoid_arm",
        title="Humanoid Arm Touch Agent",
        domain="dexterous humanoid hands and arms doing general household manipulation",
        priorities=("grasp stability", "fragile-object safety", "contact timing across fingers"),
        keywords=("humanoid", "arm", "hand", "dexterous", "household", "manipulation", "optimus", "figure"),
        examples=("Analyze episode_0040 for a humanoid arm", "How stable is the humanoid grasp on the egg?"),
    ),
    Profile(
        key="industrial_arm",
        name="dermis_industrial_arm",
        title="Industrial Manipulator Touch Agent",
        domain="industrial arms and grippers on production lines, where yield and repeatability matter",
        priorities=("first-pass yield", "repeatability across runs", "headroom before failure", "cycle time to stable hold"),
        checks=("fault", "margin", "force", "sensor", "slip_signal", "timing"),
        keywords=("industrial", "factory", "production", "line", "assembly", "manufacturing", "cobot", "gripper", "yield"),
        examples=("What's our yield on cans and bottles with SafeGrip?", "Industrial review of episode_0010"),
        objects=("can", "bottle", "cup"),
        yield_safety=0.7, min_margin=0.25,
        critical_faults=("fracture", "slip", "impact", "early_release", "overpress"),
        actions={
            "force_over_limit": "Set a hard force ceiling in the cell controller and log every cycle that touches it.",
            "fault_early_release": "Interlock the release on measured contact state; a timer-based release will drop parts at line speed.",
            "low_margin": "Margin is below production headroom (0.25): change finger pads or increase normal force before scaling up.",
        },
        fleet_first=True,
    ),
    Profile(
        key="quadruped",
        name="dermis_quadruped",
        title="Quadruped Foot-Contact Agent",
        domain="legged robots (robot dogs) sensing foot contact, foot slip and touchdown impact",
        priorities=("contact onset timing", "slip detection from vibration", "impact shocks", "sensor saturation on touchdown"),
        keywords=("quadruped", "dog", "legged", "foot", "feet", "paw", "spot", "unitree", "gait", "terrain"),
        examples=("Check slip detection for the robot dog", "Quadruped view of episode_0105"),
        applications=("quadruped",),
        critical_faults=("slip", "impact"),
        checks=("fault", "slip_signal", "timing", "sensor"),
        latency_ms=15.0,
        pod_labels={"index_tip": "front-left foot", "thumb_tip": "front-right foot",
                    "middle_tip": "rear-left foot", "thenar_apb": "rear-right foot"},
        actions={
            "fault_slip": "Foot slip: shorten stance on low-friction terrain and use the slip-band signal to trigger a step adjustment.",
            "slip_signal": "Slip-band vibration is a usable foot-slip detector: feed it to the gait controller as an early warning.",
            "rail": "Pods saturate on touchdown: lower sensor gain or add compliance in the foot so impacts stay in range.",
            "late_detection": "Touchdown is detected late: that delays stance switching. Lower the contact threshold.",
        },
        proxy_note=("No quadruped recordings yet, so this reads hand-glove recordings with the four pods mapped "
                    "to four feet. Slip, impact and contact-timing signals carry over; force levels don't."),
    ),
    Profile(
        key="prosthetic_hand",
        name="dermis_prosthetics",
        title="Prosthetic Hand Touch Agent",
        domain="prosthetic and bionic hands, where the wearer can't feel grip force",
        priorities=("never crushing", "never dropping", "a grip state the wearer can be told about"),
        keywords=("prosthetic", "prosthesis", "bionic", "amputee", "myoelectric", "wearer", "limb"),
        examples=("Would a prosthetic hand drop the cup here?", "Prosthetics review of the egg episodes"),
        objects=("egg", "cup", "can", "bottle"),
        yield_safety=0.6,
        critical_faults=("fracture", "slip", "early_release"),
        actions={
            "fault_early_release": "The hand opened mid-carry: the wearer won't notice. Require an explicit release gesture once holding.",
            "force_over_limit": "Too much force for daily-living objects: auto-limit grip and give the wearer haptic feedback near the limit.",
        },
    ),
    Profile(
        key="surgical",
        name="dermis_surgical",
        title="Surgical & Delicate Handling Agent",
        domain="surgical, lab and medical robots handling delicate items at very low force",
        priorities=("force resolution near zero", "sensor noise floor", "zero fractures"),
        checks=("force", "fault", "sensor", "margin", "slip_signal", "timing"),
        keywords=("surgical", "surgery", "medical", "lab", "tissue", "delicate", "fragile", "precision"),
        examples=("Is touch resolution good enough for delicate handling?", "Surgical review of the cup episodes"),
        objects=("egg", "cup"),
        yield_safety=0.5, min_margin=0.2,
        critical_faults=("fracture", "slip", "impact", "early_release", "overpress"),
        actions={
            "noisy": "Noise floor is too high for delicate work: shield the pods and calibrate against a load cell before any clinical claim.",
            "force_over_limit": "Exceeds a 50% safety factor on yield: delicate tasks need a much lower ceiling and a slower approach.",
        },
    ),
    Profile(
        key="warehouse",
        name="dermis_warehouse",
        title="Warehouse Pick & Place Agent",
        domain="logistics picking of mixed items, from light balls to a 520 g bottle",
        priorities=("no drops", "slip on heavy items", "place without impact"),
        checks=("fault", "margin", "slip_signal", "force", "sensor", "timing"),
        keywords=("warehouse", "logistics", "pick", "picking", "place", "sku", "bin", "fulfillment", "amazon", "package"),
        examples=("Which items do we drop most?", "Warehouse review of the bottle episodes"),
        objects=("bottle", "ball", "sponge", "can"),
        min_margin=0.2,
        critical_faults=("slip", "early_release", "impact"),
        fleet_first=True,
    ),
    Profile(
        key="food_agri",
        name="dermis_food_agri",
        title="Food & Agriculture Handling Agent",
        domain="food handling and harvesting of soft, bruisable produce (egg as the fragile-produce stand-in)",
        priorities=("no bruising or crushing", "secure hold on soft items"),
        keywords=("food", "agriculture", "farm", "harvest", "fruit", "produce", "egg", "kitchen", "cooking"),
        examples=("How close do we get to crushing the egg?", "Food handling review of episode_0064"),
        objects=("egg", "sponge", "ball"),
        yield_safety=0.6,
        critical_faults=("fracture", "slip"),
        actions={"fault_fracture": "Crushed: produce bruises below fracture force, so set the ceiling well under this and close slower."},
    ),
    Profile(
        key="eldercare",
        name="dermis_eldercare",
        title="Assistive & Care Robotics Agent",
        domain="home-assist and care robots handing over cups and bottles to people",
        priorities=("no spills or drops during hand-over", "no over-pressure", "predictable release"),
        keywords=("elder", "eldercare", "care", "assistive", "assist", "home", "hospital", "nursing", "handover", "disability"),
        examples=("Is the cup hand-over safe for a care robot?", "Care-robot review of the bottle episodes"),
        objects=("cup", "bottle", "can"),
        yield_safety=0.6, min_margin=0.2,
        critical_faults=("fracture", "slip", "early_release", "impact", "overpress"),
        actions={"fault_early_release": "Released before the person had it: hand-over must wait for the receiver's pull, sensed as shear."},
    ),
    Profile(
        key="teleop",
        name="dermis_teleop",
        title="Teleoperation & Haptic Glove Agent",
        domain="the Dermis glove itself: what an operator feels and how fast touch reaches them",
        priorities=("touch detection latency", "grasp-state recognition", "per-finger contact picture"),
        keywords=("teleop", "teleoperation", "glove", "haptic", "operator", "remote", "vr", "feedback", "latency"),
        examples=("How fast does the glove detect touch?", "Teleop review of episode_0000"),
        critical_faults=("fracture", "slip"),
        checks=("timing", "slip_signal", "fault", "force", "sensor", "margin"),
        latency_ms=20.0,
        actions={"late_detection": "Touch reaches the operator late: lower the contact threshold so haptic feedback fires at first contact."},
    ),
    Profile(
        key="sensor_qa",
        name="dermis_sensor_qa",
        title="Sensor Health & Calibration Agent",
        domain="the Dermis sensors themselves: saturation, noise, dropped frames and calibration state",
        priorities=("separating sensor faults from real grip failures", "noise vs spec", "calibration status"),
        checks=("sensor", "timing"),
        keywords=("sensor", "health", "calibration", "calibrate", "noise", "rail", "saturation", "imu", "magnetometer", "drift", "quality", "unreliable"),
        examples=("Are any readings unreliable in episode_0030?", "Sensor health check across the fleet"),
        critical_faults=(),
    ),
    Profile(
        key="data_curator",
        name="dermis_data_curator",
        title="Robot Learning Data Curator Agent",
        domain="building training sets from Dermis recordings for robot learning",
        priorities=("label balance", "coverage gaps", "what to record next", "baseline model results"),
        checks=("sensor", "timing"),
        in_matrix=False,
        keywords=("dataset", "data", "training", "train", "learning", "model", "labels", "collect", "record", "ml", "policy"),
        examples=("What data should we collect next?", "Summarize the dataset for training a slip detector"),
        critical_faults=(),
        fleet_first=True,
    ),
    Profile(
        key="data_explorer",
        name="dermis_data_explorer",
        title="Data Analyst Agent",
        domain="analysis of any dataset you bring (CSV, TSV, JSON or JSON lines by URL or pasted into the chat) "
               "and of the Dermis recordings: describe, filter, group, pivot, computed columns, correlations, charts",
        priorities=("correct numbers from the data", "the table or chart that answers the question",
                    "the reproducible pandas expression"),
        keywords=("table", "tables", "pivot", "crosstab", "groupby", "group by", "describe", "stats",
                  "statistics", "breakdown", "columns", "rows", "pandas", "dataframe", "spreadsheet", "explore",
                  "csv", "json", "dataset", "my data", "analyze this", "correlation"),
        examples=("Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
                  "Paste a CSV and ask a question about it", "Peak force by object and scenario",
                  "Which columns correlate most?"),
        checks=(), critical_faults=(), in_matrix=False, kind="explorer",
    ),
    Profile(
        key="visualizer",
        name="dermis_visualizer",
        title="Visualizer Agent",
        domain="charts of Dermis recordings: time series, overlays, histograms, scatter plots, pies, "
               "radar profiles, heatmaps and dashboards",
        priorities=("the right chart for the question", "the numbers behind every chart"),
        keywords=("chart", "charts", "plot", "graph", "visualize", "visualise", "visualization", "histogram",
                  "scatter", "pie", "donut", "radar", "heatmap", "overlay", "dashboard", "draw", "show me",
                  "compare", "versus", "trace", "over time"),
        examples=("Dashboard", "Plot episode 4", "Compare episode 4 with SafeGrip on",
                  "Histogram of peak force", "Scatter latency vs mass", "Heatmap of success", "Radar of episode 40"),
        checks=(), critical_faults=(), in_matrix=False, kind="visualizer",
    ),
]

BY_KEY = {p.key: p for p in PROFILES}
