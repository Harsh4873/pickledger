function renderIPLPrediction(data) {
  const safe = (value, fallback = '') => {
    if (value === null || value === undefined) return fallback;
    const text = String(value).trim();
    return text || fallback;
  };

  const num = (value) => {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const pct = (value) => {
    const parsed = num(value);
    if (parsed === null) return null;
    return parsed <= 1 ? parsed * 100 : parsed;
  };

  const fmtPct = (value, digits = 1) => {
    const parsed = pct(value);
    return parsed === null ? '—' : `${parsed.toFixed(digits)}%`;
  };

  const fmtNumber = (value, digits = 1) => {
    const parsed = num(value);
    return parsed === null ? '—' : parsed.toFixed(digits);
  };

  const escapeHtml = (value) => safe(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const team1 = safe(data?.team1 || data?.match?.team1 || data?.matchup?.team1 || data?.home_team, 'Team 1');
  const team2 = safe(data?.team2 || data?.match?.team2 || data?.matchup?.team2 || data?.away_team, 'Team 2');
  const matchLabel = safe(
    data?.match_label ||
      data?.match ||
      data?.fixture ||
      data?.matchup?.label ||
      `${team1} vs ${team2}`,
    `${team1} vs ${team2}`
  );
  const venue = safe(data?.venue || data?.match?.venue || data?.location, '');
  const toss = safe(data?.toss || data?.match?.toss, '');
  const confidence = safe(data?.confidence || data?.match?.confidence, 'MEDIUM').toUpperCase();
  const predictedWinner = safe(
    data?.predicted_winner ||
      data?.winner ||
      data?.prediction?.winner ||
      (pct(data?.team1_win_prob) >= pct(data?.team2_win_prob) ? team1 : team2),
    team1
  );

  const team1Prob = pct(data?.team1_win_prob ?? data?.probabilities?.team1 ?? data?.team1_probability);
  const team2Prob = pct(data?.team2_win_prob ?? data?.probabilities?.team2 ?? data?.team2_probability);
  const totalProb = (team1Prob ?? 0) + (team2Prob ?? 0);
  const team1Width = totalProb > 0 ? ((team1Prob ?? 0) / totalProb) * 100 : 50;
  const team2Width = 100 - team1Width;
  const players = Array.isArray(data?.selected_players)
    ? data.selected_players
    : Array.isArray(data?.fantasy_xi)
      ? data.fantasy_xi
      : Array.isArray(data?.players)
        ? data.players
        : [];

  const confidenceClass = confidence === 'HIGH' ? 'badge-high' : confidence === 'LOW' ? 'badge-low' : 'badge-medium';
  const winnerSide = String(predictedWinner).trim().toLowerCase() === String(team1).trim().toLowerCase() ? 'team1' : 'team2';
  const winnerState = typeof getIplWinnerActionState === 'function'
    ? getIplWinnerActionState(data)
    : { locked: false };

  const rows = players.map((player, index) => {
    const playerName = safe(player?.player_name || player?.name || player?.player, `Player ${index + 1}`);
    const playerTeam = safe(player?.team || player?.franchise || player?.squad, '');
    const role = safe(player?.role || player?.position, '');
    const decision = safe(player?.decision || player?.pick_decision || (player?.is_bet ? 'BET' : 'PASS'), 'PASS').toUpperCase();
    const fantasyPct = pct(player?.fantasy_probability_pct ?? player?.probability ?? player?.confidence);
    const score = num(player?.adjusted_score ?? player?.score ?? player?.predicted_points);
    const isCaptain = Boolean(player?.captain);
    const isViceCaptain = Boolean(player?.vice_captain);
    const rowClass = isCaptain ? 'captain-row' : isViceCaptain ? 'vc-row' : decision === 'BET' ? 'bet-row' : 'pass-row';
    const decisionClass = decision === 'BET' ? 'bet' : 'pass';
    const badgeClass = fantasyPct !== null && fantasyPct >= 70 ? 'badge-high' : fantasyPct !== null && fantasyPct >= 45 ? 'badge-medium' : 'badge-low';
    const captainBadge = isCaptain ? '<span class="decision-badge bet">C</span>' : '';
    const vcBadge = isViceCaptain ? '<span class="decision-badge bet">VC</span>' : '';
    const teamBadge = playerTeam ? `<span class="badge badge-low">${escapeHtml(playerTeam)}</span>` : '';
    const fantasyState = typeof getIplFantasyActionState === 'function'
      ? getIplFantasyActionState(data, index)
      : { locked: false };
    const fantasyButtonLabel = fantasyState.locked ? '✓ Added' : '✓';
    const fantasyButtonClass = `ipl-add-check${fantasyState.locked ? ' is-added' : ''}`;

    return `
      <tr class="${rowClass}">
        <td class="player-col">
          <div class="player-name">${escapeHtml(playerName)}</div>
          <div class="player-meta">${escapeHtml(role)}${role && playerTeam ? ' • ' : ''}${escapeHtml(playerTeam)}</div>
        </td>
        <td><span class="badge ${badgeClass}">${fmtPct(fantasyPct)}</span></td>
        <td><span class="decision-badge ${decisionClass}">${escapeHtml(decision)}</span></td>
        <td class="score-col">${fmtNumber(score)}</td>
        <td class="flags-col">${captainBadge}${vcBadge}${teamBadge}</td>
        <td class="add-col">
          <button
            class="${fantasyButtonClass}"
            type="button"
            ${fantasyState.locked ? 'disabled' : ''}
            onclick="addIplFantasyPick(${index})"
            aria-label="Add ${escapeHtml(playerName)} to Pick Log"
          >${fantasyButtonLabel}</button>
        </td>
      </tr>
    `;
  }).join('');

  const renderedRows = rows || '<tr><td colspan="6" class="ipl-empty">No fantasy XI available</td></tr>';

  const html = `
    <div class="ipl-prediction-container">
      <div class="ipl-winner-card">
        <div class="ipl-match-header">
          <div>
            <div class="ipl-match-title">${escapeHtml(matchLabel)}</div>
            <div class="ipl-match-subtitle">
              ${escapeHtml(team1)} vs ${escapeHtml(team2)}${venue ? ` • ${escapeHtml(venue)}` : ''}${toss ? ` • ${escapeHtml(toss)}` : ''}
            </div>
          </div>
          <span class="badge ${confidenceClass}">${escapeHtml(confidence)}</span>
        </div>

        <div class="winner-row">
          <div class="winner-label">
            <span class="winner-team ${winnerSide === 'team1' ? 'is-winner' : ''}">${escapeHtml(team1)}</span>
            <span class="winner-prob">${fmtPct(team1Prob)}</span>
          </div>
          <div class="prob-bar" role="img" aria-label="Win probability bar">
            <div class="prob-team1 ${winnerSide === 'team1' ? 'is-winner' : ''}" style="width:${team1Width.toFixed(2)}%"></div>
            <div class="prob-team2 ${winnerSide === 'team2' ? 'is-winner' : ''}" style="width:${team2Width.toFixed(2)}%"></div>
          </div>
          <div class="winner-label">
            <span class="winner-team ${winnerSide === 'team2' ? 'is-winner' : ''}">${escapeHtml(team2)}</span>
            <span class="winner-prob">${fmtPct(team2Prob)}</span>
          </div>
        </div>

        <div class="winner-callout">
          <span>Predicted winner: <strong>${escapeHtml(predictedWinner)}</strong></span>
          <button
            class="ipl-add-check${winnerState.locked ? ' is-added' : ''}"
            type="button"
            ${winnerState.locked ? 'disabled' : ''}
            onclick="addIplWinnerPick()"
            aria-label="Add winner pick to Pick Log"
          >${winnerState.locked ? '✓ Added' : '✓'}</button>
        </div>
      </div>

      <div class="ipl-fantasy-card">
        <div class="ipl-section-head">
          <div class="ipl-section-title">Fantasy XI</div>
          <div class="ipl-section-meta">${players.length ? `${players.length} players` : 'No players returned'}</div>
        </div>
        <div class="table-wrap">
          <table class="ipl-fantasy-table">
            <thead>
              <tr>
                <th>Player</th>
                <th>Prob</th>
                <th>Decision</th>
                <th>Score</th>
                <th>Tags</th>
                <th>Add</th>
              </tr>
            </thead>
            <tbody>
              ${renderedRows}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  if (typeof document !== 'undefined') {
    const target = document.getElementById('ipl-prediction-root');
    if (target) target.innerHTML = html;
  }

  return html;
}
