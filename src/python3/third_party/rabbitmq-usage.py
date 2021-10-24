import json

import pika


def producer(message: str):
    credentials = pika.PlainCredentials("admin", "admin")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost", credentials=credentials))
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='task_queue',
        body=message.encode('utf-8'),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))
    print(f"send {message}")
    connection.close()


def worker_for_consuming():
    credentials = pika.PlainCredentials("admin", "admin")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost",
                                                                   credentials=credentials,
                                                                   heartbeat=1800,
                                                                   blocked_connection_timeout=1800))
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)

    def callback(ch, method, properties, body):
        print(" [x] Received %s" % json.loads(body.decode("utf-8")))
        time.sleep(0.1)
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        # ch.basic_reject(delivery_tag=method.delivery_tag)

    # only取一条数据
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='task_queue', on_message_callback=callback)
    channel.start_consuming()


def worker_for_alone():
    credentials = pika.PlainCredentials("admin", "admin")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost",
                                                                   credentials=credentials,
                                                                   heartbeat=1800,
                                                                   blocked_connection_timeout=1800))
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)
    channel.basic_qos(prefetch_count=1)
    _, _, body = channel.basic_get(queue="task_queue", auto_ack=True)
    if body:
        print(json.loads(body.decode("utf-8")))
    else:
        print(body)
    channel.close()
