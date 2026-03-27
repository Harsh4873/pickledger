from data_models import Team, GameContext


# DEPRECATED - not wired to any real source. Do not call this gate until it is
# backed by actual lineup, injury, and weather providers.
class VerificationGate:
    @staticmethod
    def _raise_deprecated() -> None:
        raise NotImplementedError(
            "MLBPredictionModel.verification.VerificationGate is deprecated and is not connected "
            "to any real verification source."
        )

    @staticmethod
    def verify_game(game_ctx: GameContext):
        VerificationGate._raise_deprecated()
        
    @staticmethod
    def check_player_status(team: Team, player_name: str, check_lineup: bool = False):
        VerificationGate._raise_deprecated()
        
    @staticmethod
    def run_all_checks(game_ctx: GameContext):
        VerificationGate._raise_deprecated()
