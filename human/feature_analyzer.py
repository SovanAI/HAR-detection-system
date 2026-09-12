import json
from typing import Any, Dict, Optional

from human.state_extractor import HumanStateExtractor
from human.motion_analyzer import MotionAnalyzer
from human.posture_analyzer import PostureAnalyzer


class HumanFeatureAnalyzer:
    """
    Combines human state, motion, and posture into one
    feature representation for the future HAR engine.

    This module does not classify activities yet.
    """

    def __init__(self):
        self.state_extractor = HumanStateExtractor()
        self.motion_analyzer = MotionAnalyzer()
        self.posture_analyzer = PostureAnalyzer()

    def process_frame(
        self,
        fused_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process one fused YOLO + HMR frame.
        """

        # ----------------------------------------------------
        # HUMAN STATE
        # ----------------------------------------------------

        human_state = (
            self.state_extractor.extract_from_fused(
                fused_data
            )
        )

        # ----------------------------------------------------
        # MOTION
        # ----------------------------------------------------

        motion = self.motion_analyzer.analyze(
            human_state
        )

        # ----------------------------------------------------
        # POSTURE
        # ----------------------------------------------------

        posture = self.posture_analyzer.analyze(
            human_state
        )

        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        persons = []

        motion_persons = {
            int(person["person_id"]): person
            for person in motion.get(
                "persons",
                []
            )
        }

        posture_persons = {
            int(person["person_id"]): person
            for person in posture.get(
                "persons",
                []
            )
        }

        for person in human_state.get(
            "persons",
            []
        ):

            person_id = int(
                person["person_id"]
            )

            motion_data = motion_persons.get(
                person_id,
                {}
            )

            posture_data = posture_persons.get(
                person_id,
                {}
            )

            persons.append({
                "person_id": person_id,

                "detection": person.get(
                    "detection",
                    {}
                ),

                "position": {
                    "camera_translation": person.get(
                        "camera_translation",
                        {}
                    ),

                    "pelvis": person.get(
                        "joints_3d",
                        {}
                    ).get(
                        "pelvis"
                    )
                },

                "motion": motion_data.get(
                    "motion",
                    {}
                ),

                "posture": posture_data.get(
                    "posture",
                    {}
                ),

                "joints_3d": person.get(
                    "joints_3d",
                    {}
                ),
            })

        return {
            "frame_id": int(
                human_state["frame_id"]
            ),

            "timestamp": float(
                human_state["timestamp"]
            ),

            "model": "HumanFeatureAnalyzer",

            "persons": persons,
        }


def load_fused_file(
    path: str,
) -> Dict[str, Any]:
    """
    Load fused JSON.
    """

    with open(
        path,
        "r",
    ) as f:
        return json.load(f)


def save_features(
    features: Dict[str, Any],
    path: str,
) -> None:
    """
    Save combined HAR features.
    """

    with open(
        path,
        "w",
    ) as f:
        json.dump(
            features,
            f,
            indent=2,
        )
