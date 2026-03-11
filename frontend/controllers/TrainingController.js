export class TrainingController {
  constructor({ apiService, trainingView }) {
    this.apiService = apiService;
    this.trainingView = trainingView;
    this.pollTimer = null;
  }

  bind() {
    this.trainingView.bindActions({
      onDownload: () => this.downloadDataset(),
      onTrain: () => this.trainModel(),
      onRefresh: () => this.refreshStatus(),
    });
  }

  async refreshStatus() {
    try {
      const [dataset, training] = await Promise.all([
        this.apiService.getDatasetStatus(),
        this.apiService.getTrainingStatus(),
      ]);

      const fileCount = dataset.available ? `${dataset.files.length} arquivos CSV prontos` : 'dataset ainda nao baixado';
      this.trainingView.setMessage(`${training.stage}: ${training.message} · ${fileCount}`);
      this.trainingView.renderStatusSummary(training, dataset);

      if (training.is_running && !this.pollTimer) {
        this.startPolling();
      }

      if (training.is_running || training.stage === 'trained') {
        await this.refreshReport();
      }

      if (!training.is_running && this.pollTimer) {
        this.stopPolling();
      }
    } catch (error) {
      this.trainingView.setMessage(error.message);
    }
  }

  async refreshReport() {
    try {
      const report = await this.apiService.getTrainingReport();
      this.trainingView.renderReport(report);
    } catch {
      this.trainingView.renderReport({});
    }
  }

  async downloadDataset() {
    await this.#runAction(async () => {
      const response = await this.apiService.downloadDataset();
      this.trainingView.setMessage(`Dataset baixado: ${response.files.join(', ')}`);
    });
  }

  async trainModel() {
    await this.#runAction(async () => {
      const response = await this.apiService.trainModel();
      this.trainingView.setMessage(response.message || 'Treino iniciado.');
      if (response.accepted) {
        this.startPolling();
      }
    });
  }

  startPolling() {
    this.stopPolling();
    this.pollTimer = window.setInterval(() => {
      this.refreshStatus();
    }, 2500);
  }

  stopPolling() {
    if (this.pollTimer) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  async #runAction(callback) {
    this.trainingView.setBusy(true);
    try {
      await callback();
      await this.refreshStatus();
    } catch (error) {
      this.trainingView.setMessage(error.message);
    } finally {
      this.trainingView.setBusy(false);
    }
  }
}
