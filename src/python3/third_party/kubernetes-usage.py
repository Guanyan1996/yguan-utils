"""
!/usr/bin/python3.7
-*- coding: UTF-8 -*-
Author: https://github.com/Guanyan1996
         ┌─┐       ┌─┐
      ┌──┘ ┴───────┘ ┴──┐
      │                 │
      │       ───       │
      │  ─┬┘       └┬─  │
      │                 │
      │       ─┴─       │
      │                 │
      └───┐         ┌───┘
          │         │
          │         │
          │         │
          │         └──────────────┐
          │                        │
          │                        ├─┐
          │                        ┌─┘
          │                        │
          └─┐  ┐  ┌───────┬──┐  ┌──┘
            │ ─┤ ─┤       │ ─┤ ─┤
            └──┴──┘       └──┴──┘
                神兽保佑
                代码无BUG!

"""
from kubernetes.client import BatchV1Api, CoreV1Api, V1DeleteOptions, V1Job
from kubernetes.watch import Watch
from loguru import logger
from retrying import retry
from urllib3.exceptions import ProtocolError


class KubernetesTask(object):
    def __init__(self, job: V1Job, namespace="default", timeout_seconds=3600, tail_lines=200, api=None):
        self.job = job
        self.name = job.metadata.name
        self.backoff_limit = job.spec.backoff_limit
        self.timeout_seconds = timeout_seconds
        self.ns = namespace
        self.corev1api = CoreV1Api()
        self.tail_lines = tail_lines

        if not api:
            self.api = BatchV1Api()
        else:
            self.api = api

    def create_job(self):
        response = self.api.create_namespaced_job(body=self.job, namespace=self.ns)
        logger.info(
            f"EVENT: JOB({self.name}) has been created in {self.ns} namespace with backoff_limit {self.backoff_limit}")
        return response

    def delete_job(self):
        response = self.api.delete_namespaced_job(
            name=self.name,
            namespace=self.ns,
            body=V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5))
        logger.info(f"Job {self.name} deleted in {self.ns}")
        return response

    def get_job_log(self):
        pod_list = self.corev1api.list_namespaced_pod(namespace=self.ns, label_selector=f"job-name={self.name}")
        for pod in pod_list.items:
            pod_logger = self.corev1api.read_namespaced_pod_log(namespace=self.ns, name=pod.metadata.name,
                                                                tail_lines=self.tail_lines)
            logger.info(
                f"JOB({self.name})'s POD({pod.metadata.name}) loggers \n {pod_logger}")

    # https://v1-18.docs.kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#jobcondition-v1-batch
    @retry(stop_max_attempt_number=6)
    def wait_util_completed(self):
        w = Watch()
        try:
            for event in w.stream(self.api.list_namespaced_job,
                                  namespace=self.ns,
                                  label_selector=f"job-name={self.name}",
                                  timeout_seconds=self.timeout_seconds):
                # issue https://github.com/kubernetes-client/python/issues/540
                job = event["object"]
                status = job.status
                condition1 = status.succeeded is not None
                condition2 = int(status.succeeded or 0) + int(status.failed or 0) == self.backoff_limit + 1
                condition3 = event['type'] == "DELETED"
                event_job_name = event['object'].metadata.name
                logger.info(
                    f"{event['object'].kind}({event_job_name}) now status is {event['type']} succeeded({status.succeeded}), failed({status.failed})")
                if condition1 or condition2 or condition3:
                    # self.get_job_loggers()
                    w.stop()
                    logger.info(
                        f"{event['object'].kind}({event_job_name}) watch monitor is stopped")
                    return {
                        f"{event_job_name}": {
                            "namespace": self.ns,
                            "succeeded": status.succeeded,
                            "failed": status.failed
                        }
                    }
        except ProtocolError:
            logger.warning('skip this error... because kubelet disconnect client after default 10m...')

    def run(self):
        self.create_job()
        return self.wait_util_completed()
