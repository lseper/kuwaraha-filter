import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";

interface Props {
	beforeImageUrl: string;
	afterImageUrl?: string;
}

function BeforeAndAfterImages({ beforeImageUrl, afterImageUrl }: Props) {
	return (
		<Container
			sx={{
				display: "flex",
				justifyContent: "center",
				alignItems: "center",
				marginTop: "16px",
				textAlign: "center",
			}}
		>
			<Card>
				<CardContent>
					<Typography variant="h6" align="center">
						Input
					</Typography>
					<img
						src={beforeImageUrl}
						alt="before"
						style={{
							width: "100%",
							height: "auto",
							marginRight: "8px",
						}}
					/>
				</CardContent>
			</Card>
			<Card>
				<CardContent>
					<Typography variant="h6" align="center">
						Result
					</Typography>
					{afterImageUrl && (
						<img
							src={afterImageUrl}
							alt="after"
							style={{
								width: "100%",
								height: "auto",
								marginRight: "8px",
							}}
						/>
					)}
				</CardContent>
			</Card>
		</Container>
	);
}

export default BeforeAndAfterImages;
