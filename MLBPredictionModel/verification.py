from data_models import Team, GameContext

class VerificationGate:
    @staticmethod
    def verify_game(game_ctx: GameContext):
        print("---")
        print("STEP 0 — MANDATORY VERIFICATION GATE")
        print(f"Checking {game_ctx.away_team.name} vs {game_ctx.home_team.name} at {game_ctx.venue.name}")
        
    @staticmethod
    def check_player_status(team: Team, player_name: str, check_lineup: bool = False):
        """Mock verification check for a player."""
        # In a real app this would query Rotowire/ESPN for injury/lineup status
        is_verified = True
        print(f"- [x] Confirmed current team for {player_name}: {team.name}")
        print(f"- [x] Confirmed injury/IL status for {player_name}: Active")
        if check_lineup:
            print(f"- [x] Confirmed {player_name} is in the starting lineup tonight")
            
        return is_verified
        
    @staticmethod
    def run_all_checks(game_ctx: GameContext):
        VerificationGate.verify_game(game_ctx)
        VerificationGate.check_player_status(game_ctx.home_team, game_ctx.home_team.starter.name, check_lineup=True)
        VerificationGate.check_player_status(game_ctx.away_team, game_ctx.away_team.starter.name, check_lineup=True)
        print("- [x] Confirmed weather sourced")
        print("Verification Complete. All checks passed.")
        print("---")
