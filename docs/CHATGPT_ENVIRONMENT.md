# Save, Publish and GitHub are separate actions

In this ChatGPT cloud-environment setup workflow:

- **Save** stores the editable environment configuration/draft so you can continue reviewing or changing it. It does not make a Git commit or push a repository.
- **Publish** makes that environment version reusable, including its configured setup and filesystem snapshot. Future environments can start from that version. A new draft can then be based on the latest published version, as the chat messages describe. Running processes are not a saved Colab session.
- **Git commit / push** records repository changes and sends commits to a GitHub branch. Environment Save/Publish does not substitute for these actions.
- **Save in Colab** saves a notebook. Video artifacts persist only where the code writes them, such as the mounted Drive folder; Colab's local disk is temporary.

This work does not publish a ChatGPT environment or publish a video. Repository changes are reviewed independently through the feature branch. Keep credentials in the environment's secret configuration or Colab Secrets, not in files intended for a snapshot or GitHub.
