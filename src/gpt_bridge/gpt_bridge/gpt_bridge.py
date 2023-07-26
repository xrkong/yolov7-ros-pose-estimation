import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class GptNode(Node):
    def __init__(self):
        super().__init__('gpt_node')
        self.publisher_ = self.create_publisher(String, 'gpt_response', 10)
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        prompt = msg.data
        self.get_logger().info('I heard: "%s"' % prompt)

        # Call GPT API
        # response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        # messages=[
        #     {"role": "system", "content": prompt}
        # ]
        # )

        # res_msg = String()
        # res_msg.data = response['choices'][0]['message']['content']
        # self.publisher_.publish(res_msg)
        # print(res_msg.data)

def main(args=None):
    rclpy.init(args=args)

    gpt_node = GptNode()

    rclpy.spin(gpt_node)

    # Destroy the node explicitly
    gpt_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
