import os

class Logger:
    def __init__(self, log_level, **kwargs):

        if log_level is not None:
           os.environ['PREFECT_LOGGING_LEVEL'] = log_level


        # we delay the loading prefect modules so that we can set the LOGGING_LEVEL

        from prefect.context import get_run_context
        from prefect.logging import get_run_logger

        self.info = Logger._prefect_logger("info", get_run_logger)
        self.debug = Logger._prefect_logger("debug", get_run_logger)
        self.warning = Logger._prefect_logger("warning", get_run_logger)
        self.error = Logger._prefect_logger("error", get_run_logger)


        from prefect.artifacts import (
            create_markdown_artifact,
            create_link_artifact,
            create_table_artifact,
            create_image_artifact,
            create_progress_artifact,
            update_progress_artifact
        )

        self.create_markdown_artifact = create_markdown_artifact
        self.create_link_artifact = create_link_artifact
        self.create_table_artifact = create_table_artifact
        self.create_image_artifact = create_image_artifact
        self.create_progress_artifact = create_progress_artifact
        self.update_progress_artifact = update_progress_artifact

    @staticmethod
    def _prefect_logger(level: str, get_run_logger):
        def log_fn(msg, *args, **kwargs):
            logger = get_run_logger()
            return getattr(logger, level)(msg, *args, **kwargs)
        return log_fn


    def artifact(self, **kwargs):
        """
        Create a Prefect artifact based on keyword inputs:
        - markdown
        - link (+ optional text)
        - table
        - image (+ auto-detect image_type from file extension)
        Optional shared keys: key, description
        """


        if "markdown" in kwargs:
            return self.create_markdown_artifact(
                markdown=kwargs["markdown"],
                key=kwargs.get("key"),
                description=kwargs.get("description"),
            )

        elif "link" in kwargs:
            return self.create_link_artifact(
                link_url=kwargs["link"],
                link_text=kwargs.get("text") or kwargs["link"],
                key=kwargs.get("key"),
                description=kwargs.get("description"),
            )

        elif "table" in kwargs:
            return self.create_table_artifact(
                table=kwargs["table"],
                key=kwargs.get("key"),
                description=kwargs.get("description"),
            )

        elif "image" in kwargs:
            image = kwargs["image"]

            return self.create_image_artifact(
                image_url=image,
                key=kwargs.get("key"),
                description=kwargs.get("description"),
            )
        elif "progress" in kwargs:
            if "progress_id" not in kwargs:
                # If no progress_id is provided, we create a new progress artifact
                return self.create_progress_artifact(
                    progress=kwargs["progress"],
                    description=kwargs.get("description"),
                )
            else:
                return self.update_progress_artifact(
                    artifact_id=kwargs["progress_id"],
                    progress=kwargs["progress"]
                )
        else:
            raise ValueError("No valid artifact type specified. Use one of: markdown, link, table, image.")




