export class TrainingView {
  constructor() {
    this.statusMessage = document.querySelector('#training-status-message');
    this.downloadButton = document.querySelector('#download-dataset-button');
    this.trainButton = document.querySelector('#train-model-button');
    this.refreshButton = document.querySelector('#refresh-status-button');
    this.kpis = document.querySelector('#training-kpis');
    this.summary = document.querySelector('#training-summary');
    this.rankingMetrics = document.querySelector('#training-ranking-metrics');
    this.lossChart = document.querySelector('#loss-chart');
    this.validationChart = document.querySelector('#validation-chart');
    this.textWeights = document.querySelector('#text-weights');
    this.numericWeights = document.querySelector('#numeric-weights');
    this.hyperparameters = document.querySelector('#training-hyperparameters');
  }

  bindActions({ onDownload, onTrain, onRefresh }) {
    this.downloadButton?.addEventListener('click', onDownload);
    this.trainButton?.addEventListener('click', onTrain);
    this.refreshButton?.addEventListener('click', onRefresh);
  }

  setMessage(message) {
    if (this.statusMessage) {
      this.statusMessage.textContent = message;
    }
  }

  setBusy(isBusy) {
    [this.downloadButton, this.trainButton, this.refreshButton].forEach((button) => {
      if (button) {
        button.disabled = isBusy;
      }
    });
  }

  renderStatusSummary(status = {}, dataset = {}) {
    if (!this.kpis) {
      return;
    }

    const cards = [
      { label: 'Estagio', value: status.stage || 'desconhecido' },
      { label: 'Dataset', value: dataset.available ? `${dataset.files.length} arquivos` : 'nao baixado' },
      { label: 'Execucao', value: status.is_running ? 'treinando' : 'parado' },
      { label: 'Progresso', value: `${status.progress ?? 0}%` },
      { label: 'Loss atual', value: typeof status.loss === 'number' ? status.loss.toFixed(4) : '-' },
      { label: 'Val loss', value: typeof status.val_loss === 'number' ? status.val_loss.toFixed(4) : '-' },
    ];

    this.kpis.innerHTML = cards.map((card) => `
      <div class="kpi-card">
        <strong>${card.label}</strong>
        <span>${card.value}</span>
      </div>
    `).join('');
  }

  renderReport(report = {}) {
    const trainingSummary = report.training_summary || {};
    const trainingReport = report.training_report || {};
    const rankingMetrics = report.ranking_metrics || {};
    const hyperparameters = report.hyperparameters || {};
    const textWeights = report.text_feature_config || {};
    const numericWeights = report.numeric_feature_weights || {};

    if (this.summary) {
      const estimatedPrecision = this.#estimatePrecision(trainingSummary, trainingReport);
      const summaryItems = [
        ['Filmes', trainingSummary.movies],
        ['Loss final', this.#formatNumber(trainingSummary.autoencoder_loss)],
        ['Epocas', trainingSummary.epochs_ran],
        ['Melhor validacao', this.#formatNumber(trainingSummary.best_validation_score)],
        ['Precisao estimada', estimatedPrecision],
        ['Usuarios avaliados', rankingMetrics.evaluated_users ?? '-'],
        ['P@5 offline', this.#formatPercent(rankingMetrics.precision_at_5)],
        ['Recall@5 offline', this.#formatPercent(rankingMetrics.recall_at_5)],
        ['NDCG@5 offline', this.#formatPercent(rankingMetrics.ndcg_at_5)],
        ['MRR offline', this.#formatPercent(rankingMetrics.mrr)],
        ['Pontos na curva', trainingSummary.loss_curve_length],
        ['Bottleneck', hyperparameters.bottleneck_dimensions],
      ];

      this.summary.innerHTML = summaryItems.map(([label, value]) => `
        <div class="parameter-item">
          <strong>${label}</strong>
          <span>${value ?? '-'}</span>
        </div>
      `).join('');
    }

    this.#renderRankingMetrics(rankingMetrics);

    this.#renderChart(this.lossChart, trainingReport.loss_curve || [], 'loss');
    this.#renderChart(this.validationChart, trainingReport.validation_scores || [], 'validation');
    this.#renderParameters(this.textWeights, textWeights, true);
    this.#renderParameters(this.numericWeights, numericWeights, false);
    this.#renderParameters(this.hyperparameters, hyperparameters, false);
  }

  #renderParameters(container, data, isNested) {
    if (!container) {
      return;
    }
    const entries = Object.entries(data || {});
    if (!entries.length) {
      container.innerHTML = '<div class="parameter-item">Sem dados ainda.</div>';
      return;
    }

    container.innerHTML = entries.map(([key, value]) => {
      const printable = isNested
        ? Object.entries(value).map(([childKey, childValue]) => `${childKey}: ${Array.isArray(childValue) ? childValue.join(' x ') : childValue}`).join(' · ')
        : `${Array.isArray(value) ? value.join(' x ') : value}`;

      return `
        <div class="parameter-item">
          <strong>${key}</strong>
          <span>${printable}</span>
        </div>
      `;
    }).join('');
  }

  #renderChart(container, values, variant) {
    if (!container) {
      return;
    }
    if (!values.length) {
      container.classList.add('empty-chart');
      container.textContent = 'Sem dados de treino ainda.';
      return;
    }

    container.classList.remove('empty-chart');
    const width = 520;
    const height = 220;
    const padding = 20;
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const denominator = maxValue - minValue || 1;

    const points = values.map((value, index) => {
      const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
      const y = height - padding - ((value - minValue) / denominator) * (height - padding * 2);
      return `${x},${y}`;
    }).join(' ');

    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="grafico ${variant}">
        <polyline class="chart-line ${variant}" points="${points}"></polyline>
      </svg>
    `;
  }

  #renderRankingMetrics(rankingMetrics) {
    if (!this.rankingMetrics) {
      return;
    }

    if (!rankingMetrics.available) {
      this.rankingMetrics.innerHTML = `<div class="parameter-item">${rankingMetrics.reason || 'Sem avaliacao offline ainda.'}</div>`;
      return;
    }

    const items = [
      ['Usuarios avaliados', rankingMetrics.evaluated_users],
      ['Precision@5', this.#formatPercent(rankingMetrics.precision_at_5)],
      ['Recall@5', this.#formatPercent(rankingMetrics.recall_at_5)],
      ['NDCG@5', this.#formatPercent(rankingMetrics.ndcg_at_5)],
      ['Precision@10', this.#formatPercent(rankingMetrics.precision_at_10)],
      ['Recall@10', this.#formatPercent(rankingMetrics.recall_at_10)],
      ['NDCG@10', this.#formatPercent(rankingMetrics.ndcg_at_10)],
      ['MRR', this.#formatPercent(rankingMetrics.mrr)],
    ];

    this.rankingMetrics.innerHTML = items.map(([label, value]) => `
      <div class="parameter-item">
        <strong>${label}</strong>
        <span>${value ?? '-'}</span>
      </div>
    `).join('');
  }

  #formatNumber(value) {
    return typeof value === 'number' ? value.toFixed(4) : '-';
  }

  #formatPercent(value) {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '-';
  }

  #estimatePrecision(trainingSummary, trainingReport) {
    const finalValidation = trainingSummary.final_validation_score
      ?? trainingReport.validation_scores?.filter((value) => typeof value === 'number').at(-1)
      ?? trainingSummary.best_validation_score;

    return typeof finalValidation === 'number'
      ? `${(finalValidation * 100).toFixed(1)}%`
      : '-';
  }
}
